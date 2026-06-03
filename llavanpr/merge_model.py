from PIL import Image
import requests
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoConfig
from modeling_llavanpr import CustomLlavaNextForConditionalGeneration, ResamplerConfig, PerceiverResampler, LlavaNextMultiModalProjector, OTFusion
from config_llavanpr import LlavaWithVisionExpertConfig, ResnetConfig
from processing_llavanpr import CustomLlavaNextProcessor
import torch
import copy

if __name__ == '__main__':

    resnet50_config = ResnetConfig()

    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    #import requests

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    image_path = r'../VLMEvalKit/scripts/real.jpg'
    image = Image.open(image_path)

    processor = CustomLlavaNextProcessor.from_pretrained(r'llava-hf/llava-v1.6-mistral-7b-hf')
    
    config1 = AutoConfig.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    resampler_config = ResamplerConfig()
    resampler_config.num_resampler_layers = 3
    resampler_config.hidden_size = config1.text_config.hidden_size
    resampler_config.vision_hidden_size = 512
    config = LlavaWithVisionExpertConfig(expert_config=resnet50_config, resampler_config=resampler_config, **config1.to_dict())
    config.vision_feature_select_strategy = "expert"
    #config.vision_config.hidden_size=512
    model = CustomLlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16,config=config)
    config.model_type = 'llava_with_vision_expert'
    processor = CustomLlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model.vision_tower_expert.model.load_state_dict(torch.load(r'https://drive.google.com/drive/folders/1_mD17F94xMbJqEAsWRW1gVsZ5db6YamI?usp=sharing'), strict=False)
    expert_projector_config = copy.deepcopy(config)
    expert_projector_config.vision_config.hidden_size=512
    multi_modal_expert_projector = LlavaNextMultiModalProjector(expert_projector_config)
    otfusion = OTFusion(resampler_config)
    #### initialize expert projector
    model.multi_modal_expert_projector = copy.deepcopy(multi_modal_expert_projector)
    model.multi_modal_expert_projector_otfusion = copy.deepcopy(otfusion)

    save_path = "./checkpoints/TranXAdapter-LLaVA-next-mistral7B-v0"
    model = model.to(torch.float16)
    model.save_pretrained(save_path)
    processor.num_additional_image_tokens = 0
    processor.save_pretrained(save_path)
    print(model)
