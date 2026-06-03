from typing import List, Union
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers.models.qwen3_vl import Qwen3VLProcessor
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import Unpack
from transformers import BatchFeature
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs
import math

def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img

def crop(img):
    img_array = np.array(img)
    w, h = img.size
    short_edge = min(w, h)
    crop_func = transforms.CenterCrop(short_edge)
    crop_img = crop_func(img)
    
    return crop_img


class CustomQwen3VLProcessor(Qwen3VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs):
        
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.vlm_size = getattr(image_processor, "vlm_size", 256)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: crop(img)),
            transforms.Resize([self.vlm_size, self.vlm_size])
            
        ])
        self.npr_transform = transforms.Compose([
            transforms.Lambda(lambda img: translate_duplicate(img, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __call__(self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
        ):
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            vit_images = [self.transform(image) for image in images]
            image_inputs = self.image_processor(images=vit_images, **output_kwargs["images_kwargs"])
            npr_image_tensor = [self.npr_transform(image) for image in images]
            npr_image_tensor = torch.stack(npr_image_tensor,dim=0)
            image_inputs['npr_pixel_values'] = npr_image_tensor
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
            
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    num_image_tokens += 196
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)
                
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        # logger.warning_once(
                        #     "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                        #     "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                        #     "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        # )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    # if timestamps are not provided, calculate them
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)
