from PIL import Image
from transformers import AutoConfig
from modeling_qwen3vlnpr import CustomQwen3VLForConditionalGeneration, Qwen3VLMultiModalExpertProjector, OTFusion, ResamplerConfig
from config_qwen3vlnpr import Qwen3VLWithVisionExpertConfig, ResnetConfig
from processing_qwen3vlnpr import CustomQwen3VLProcessor
import torch
import copy
import os
from torchvision import transforms
from qwen_vl_utils import process_vision_info

def test(model, processor):
    print(model)
    model = model.cuda(0)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file://../VLMEvalKit/scripts/fake.jpg",
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 256 * 32 * 32,
                },
                {"type": "text", "text": "Please describe the image."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

    # split the videos and according metadatas
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        #videos[0] = videos[0][:9]
        to_pil = transforms.ToPILImage()    
    else:
        video_metadatas = None

    # since qwen-vl-utils has resize the images/videos, \
    # we should pass do_resize=False to avoid duplicate operation in processor!
    inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # debug = 0
    generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.2)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


if __name__ == '__main__':
    original_model_path = r'Qwen/Qwen3-VL-4B-Instruct'
    artifact_model_path = r'https://drive.google.com/drive/folders/1_mD17F94xMbJqEAsWRW1gVsZ5db6YamI?usp=sharing'
    save_path = r'./checkpoints'
    os.makedirs(save_path, exist_ok=True)
    
    resnet50_config = ResnetConfig()
    image_path = r'../VLMEvalKit/scripts/fake.jpg'
    image = Image.open(image_path)
    processor = CustomQwen3VLProcessor.from_pretrained(original_model_path)
    
    config1 = AutoConfig.from_pretrained(original_model_path)
    config = Qwen3VLWithVisionExpertConfig(expert_config=resnet50_config, **config1.to_dict())
    model = CustomQwen3VLForConditionalGeneration.from_pretrained(original_model_path, torch_dtype=torch.float16, config=config)
    config.model_type = 'qwen3vl_with_vision_expert'
    model.model.vision_tower_expert.model.load_state_dict(torch.load(artifact_model_path), strict=False)
    
    expert_config = copy.deepcopy(config)
    expert_config.vision_config.hidden_size = 512
    multi_modal_expert_projector = Qwen3VLMultiModalExpertProjector(expert_config.vision_config)
    model.model.multi_modal_expert_projector = copy.deepcopy(multi_modal_expert_projector)
    
    resampler_config = ResamplerConfig()
    resampler_config.hidden_size = config.text_config.hidden_size
    resampler_config.vision_hidden_size = 512
    otfusion = OTFusion(resampler_config)
    model.model.multi_modal_expert_projector_otfusion = copy.deepcopy(otfusion)
    
    model = model.to(torch.float16)
    test(model, processor)
    
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
