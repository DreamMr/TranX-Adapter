# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from swift.llm import to_device, to_float_dtype
from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio, load_batch, load_video_ovis2
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta, ThinkingTemplate
import numpy as np
from PIL import Image
from random import randint
import torchvision.transforms as transforms
import kornia.augmentation as K
import kornia

@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


qwq_preview_system = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                      'You should think step-by-step.')

register_template(QwenTemplateMeta(LLMTemplateType.qwen))
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))
register_template(QwenTemplateMeta(LLMTemplateType.qwq_preview, default_system=qwq_preview_system))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwq, default_system=None, response_prefix='<think>\n', template_cls=ThinkingTemplate))

# '<think>\n\n</think>\n\n'
register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=ThinkingTemplate))


class Qwen3RerankerTemplate(Template):
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        query = inputs.messages[-2]['content']
        user_message = '<Instruct>: ' + self.instruction + '\n' + '<Query>: ' + query + '\n' + '<Document>: {doc}'
        inputs.messages[-2]['content'] = user_message


qwen3_reranker_system = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be \"yes\" or \"no\".')

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_reranker,
        default_system=qwen3_reranker_system,
        response_prefix='<think>\n\n</think>\n\n',
        template_cls=Qwen3RerankerTemplate))

register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenPRMTemplate(Template):
    cot_process_placeholder = '<extra_0>'

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        super()._preprocess_inputs(inputs)
        total_content = '\n'.join([message['content'] or '' for message in inputs.messages])
        if self.cot_process_placeholder not in total_content:
            inputs.messages[-1]['content'] = inputs.messages[-1]['content'] + self.cot_process_placeholder

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        step_sep_id = self.tokenizer.encode(self.cot_process_placeholder)[0]
        token_masks = (input_ids == step_sep_id)
        return self.make_step_rewards(logits, token_masks)


register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math_prm, template_cls=QwenPRMTemplate))


class QwenVLTemplate(Template):
    load_images = False

    @staticmethod
    def _load_image(image, load_images: bool):
        if not load_images and isinstance(image, str) and (image.startswith('data:') or len(image) > 200):
            load_images = True
        return Template._load_image(image, load_images)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'lmdeploy':
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            image = inputs.images[index]
            if self.mode == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>{self._get_bbox_str(bbox)}</box>']


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **tokenizer_kwargs):
        audio_info = self.processor.process_audio(context)
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        audio_info = self.processor.process_audio(text)
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            video = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                    media_inputs = processor_func(images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        from qwen_vl_utils import vision_process
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / vision_process.FPS
                        ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                loss_scale = self._extend_loss_scale(loss_scale, idx_list, _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def forward_context(self, model, inputs):
        if 'real_position_ids' not in inputs:
            return super().forward_context(model, inputs)
        if self.version == 'v2':
            from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling_module
        elif self.version == 'v2_5':
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling_module
        elif self.version == 'omni':
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        position_ids = inputs['position_ids']
        inputs['position_ids'] = inputs.pop('real_position_ids')
        return self._patch_flash_attention_forward(modeling_module, position_ids)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')

        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            mixed_embeds = model.visual(pixel_values_mixed, grid_thw=grid_thw)
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = self.processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            if image_embeds is not None:
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw
        return res

    def packing_row(self, row: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        position_ids = []
        for r in row:
            r = r[0].copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['real_position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_position_ids(self, inputs: Dict[str, Any]):
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        base_model = self.get_base_model(self.model)
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        else:
            get_rope_index = base_model.model.get_rope_index
        position_ids, _ = get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask=inputs.get('attention_mask'),
            **kwargs)
        return position_ids.contiguous()

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if self._packing:
            res['real_position_ids'] = self.concat_tensor(batch, 'real_position_ids', -1)
        elif self.is_training:
            res['position_ids'] = self._get_position_ids(res)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
    ))


class Qwen2_5VLTemplate(Qwen2VLTemplate):
    version = 'v2_5'
    norm_bbox = 'none'


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_vl, template_cls=Qwen2_5VLTemplate))

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

from io import BytesIO
from random import random, choice, randint
import cv2
from scipy.ndimage.filters import gaussian_filter

def crop(img):
    img_array = np.array(img)
    w, h, c = img_array.shape
    short_edge = min(w, h)
    if short_edge == w:
        choice_point = randint(0, h - short_edge)
        crop_image = img_array[:w, choice_point:choice_point + short_edge, :]
    else:
        choice_point = randint(0, w - short_edge)
        crop_image = img_array[choice_point:choice_point + short_edge, :h, :]
    crop_image = Image.fromarray(crop_image)
    # resize_img = img_array[:short_edge, :short_edge, :]
    # resize_img = Image.fromarray(resize_img)
    return crop_image

# def pil_to_tensor(img):
#     img = np.array(img)
#     return kornia.image_to_tensor(img, keepdim=True).float() / 255.0


def pil_jpg(img, qual=75):
    out = BytesIO()
    #img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=qual)
    img = Image.open(out)
    img.load()
    # load from memory before ByteIO closes
    # img = np.array(img)
    out.close()
    return img

def pil_png(img, compress_val=6):
    """
    将 numpy 数组或 PIL.Image 转换为 PNG 格式，并返回 PIL.Image。

    参数:
        img: numpy.ndarray | PIL.Image
        compress_val: int, PNG 压缩等级，范围 0-9，默认 6。
                      0 压缩最少，9 压缩最多。
    """
    out = BytesIO()
    #img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    img.save(out, format='PNG', compress_level=compress_val)
    img = Image.open(out)
    img.load()
    out.close()
    return img

def center_crop(img, crop_size=(128, 128)):
    # img: HxWxC or HxW
    h, w = img.shape[:2]
    th, tw = crop_size
    
    # 确保图片足够大，否则需要先 padding (这里略过 padding 逻辑)
    if h < th or w < tw:
        #raise ValueError(f"图片尺寸 ({h},{w}) 小于目标尺寸 ({th},{tw})")
        return img
        
    # 计算左上角起始点
    y_start = (h - th) // 2
    x_start = (w - tw) // 2
    
    # 利用 numpy 切片进行裁剪
    return img[y_start : y_start+th, x_start : x_start+tw]

def data_augment(img, blur_prob, blur_sig, jpg_prob, jpg_qual, jpg_method, resolu_prob=0.5):
    img = np.array(img)

    if random.random() < blur_prob:
        sig = sample_continuous(blur_sig)
        gaussian_blur(img, sig)

    if random.random() < jpg_prob:
        method = sample_discrete(jpg_method)
        qual = sample_discrete(jpg_qual)
        img = jpeg_from_key(img, qual, method)
    
    # if random.random() < resolu_prob:
    #     img = center_crop(img, crop_size=(128,128))

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

import random
def convert_format(img):
    if random.random() > 0.5:
        img_format = img.format
        if img_format == 'JPEG':
            return pil_png(img)
        else:
            return pil_jpg(img)
    else:
        return img

def crop(img):
    img_array = np.array(img)
    w, h, c = img_array.shape
    short_edge = min(w, h)
    if short_edge == w:
        choice_point = randint(0, h - short_edge)
        crop_image = img_array[:w, choice_point:choice_point + short_edge, :]
    else:
        choice_point = randint(0, w - short_edge)
        crop_image = img_array[choice_point:choice_point + short_edge, :h, :]
    crop_image = Image.fromarray(crop_image)
    # resize_img = img_array[:short_edge, :short_edge, :]
    # resize_img = Image.fromarray(resize_img)
    return crop_image

class Qwen3VLTemplate(Qwen2VLTemplate):
    version = 'v3'

    
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from qwen3vlnpr.processing_qwen3vlnpr import CustomQwen3VLProcessor
        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        
        
        blur_prob = 0.1
        jpg_prob = 0.1
        blur_sig=[0.0,3.0]
        jpg_qual = list(range(30,100+1)) # [30,100]
        jpg_method = ['cv2', 'pil']
        Perturbations = transforms.Compose([
            transforms.Lambda(lambda img: data_augment(img, blur_prob=blur_prob, blur_sig=blur_sig, jpg_prob=jpg_prob, jpg_qual=jpg_qual, jpg_method=jpg_method))
        ])
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: crop(img)),
            transforms.Resize([256, 256]) # RRDataset: [512, 512]
        ])
        self.npr_transform = transforms.Compose([
            transforms.Lambda(lambda img: translate_duplicate(img, 224)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.vit_resize = K.Resize((256, 256))
        self.npr_crop = K.CenterCrop((224, 224))
        self.npr_resize = K.Resize((384,384))
        self.npr_norm = K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

        #print("start data")
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    mm_data = [Perturbations(image) for image in mm_data]
                    clip_images = [self.transform(img) for img in mm_data]
                    media_inputs = processor.image_processor(images=clip_images, return_tensors='pt', do_resize=False)
                    #media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                    additional_image_tokens = 0
                    if isinstance(processor, CustomQwen3VLProcessor):
                        npr_image_tensor = [self.npr_transform(image) for image in mm_data]
                        npr_image_tensor = torch.stack(npr_image_tensor,dim=0)
                        media_inputs['npr_pixel_values'] = npr_image_tensor
                        additional_image_tokens = 196 # for npr
                else:
                    split_token = self._tokenize('\n')[0]
                    media_inputs = processor(
                        text=['\n'.join(['<|vision_start|><|video_pad|><|vision_end|>'] * len(mm_data))],
                        videos=mm_data,
                        return_tensors='pt',
                        do_resize=False,
                        **inputs.mm_processor_kwargs)
                    splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_inputs.pop('input_ids', None)
                    media_inputs.pop('attention_mask', None)
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == 'images':
                        token_len = (media_grid_thw[i].prod() // merge_length) + additional_image_tokens
                        return [media_token] * token_len
                    else:
                        return splited_tokens[i]

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list,
                                                                    _get_new_tokens)
                loss_scale = self._extend_loss_scale(loss_scale, idx_list, _get_new_tokens)
                encoded.update(media_inputs)
        #print("finish data")
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qwen3_vl, template_cls=Qwen3VLTemplate, default_system=None, thinking_prefix='<think>\n'))


register_template(
    QwenTemplateMeta(
        MLLMTemplateType.mimo_vl,
        template_cls=Qwen2_5VLTemplate,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.'))


class Qwen2_5OmniTemplate(Qwen2_5VLTemplate):
    version = 'omni'
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        if processor is None:
            return
        super().init_processor(processor)
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        default = Qwen2_5OmniProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            inputs.videos[index] = fetch_video({'video': video}).to(torch.uint8)
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            return ['<|vision_bos|><|VIDEO|><|vision_eos|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        # audio
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = (((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1)
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        if idx_list:
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_audio_tokens)
            loss_scale = self._extend_loss_scale(loss_scale, idx_list, _get_new_audio_tokens)

        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']

                    def _get_new_tokens_use_audio_in_video(i):
                        audio_token_indices = torch.arange(audio_lengths[i])
                        grid_thw = media_grid_thw[i]
                        height = grid_thw[1] // merge_size
                        width = grid_thw[2] // merge_size
                        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = torch.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)).reshape(-1)
                        video_token_indices = (
                            video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
                        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
                        video_chunk_indexes = processor.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        res = []
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                res += token_id * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                res += audio_token_id * audio_seq_length
                        return res

                    input_ids, labels = self._extend_tokens(input_ids, labels, idx_list,
                                                            _get_new_tokens_use_audio_in_video)
                    loss_scale = self._extend_loss_scale(loss_scale, idx_list, _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                    loss_scale = self._extend_loss_scale(loss_scale, idx_list, _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded.update(media_inputs)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return Template._post_encode(self, model, inputs)

    def _get_position_ids(self, inputs: Dict[str, Any]):
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return position_ids.contiguous()

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_omni, template_cls=Qwen2_5OmniTemplate))


class Ovis1_6Template(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            max_partition = get_env_args('max_partition', int, 9)
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        dtype = self.model.visual_tokenizer.dtype
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype)
        else:
            pixel_values = torch.zeros((1, 3, 384, 384), dtype=dtype)  # dummpy
        encoded.update({'input_ids': input_ids, 'labels': labels})
        encoded['pixel_values'] = [pixel_values]
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        padding_side = self.padding_side if self.is_training else 'left'
        if self.max_length is not None:
            model.config.multimodal_max_length = self.max_length
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if labels is None:
            labels = input_ids.new_full(input_ids.shape, -100)
        _, inputs_embeds, labels, attention_mask = model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=torch.ones_like(input_ids),  # not use, only compat
            text_labels=labels,
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')
        if inputs.get('labels') is None:
            labels = None
        return {'inputs_embeds': inputs_embeds, 'labels': labels, 'attention_mask': attention_mask}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.ovis1_6,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        template_cls=Ovis1_6Template,
    ))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.ovis1_6_llama3,
        default_system='You are a helpful and honest multimodal assistant.',
        template_cls=Ovis1_6Template,
    ))


class Ovis2Template(Ovis1_6Template):
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    nframes = 12

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image>\n']
            return [[-200], '\n']
        elif media_type == 'video':
            nframes = get_env_args('nframes', int, self.nframes)
            inputs.images = load_video_ovis2(inputs.videos[index], nframes)
            return [[-200] * nframes, '\n']


register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2,
    template_cls=Ovis2Template,
))


@dataclass
class MarcoO1TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = """
你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.
        \n## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
        """


register_template(MarcoO1TemplateMeta(LLMTemplateType.marco_o1))
