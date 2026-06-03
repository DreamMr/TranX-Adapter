from transformers import LlavaConfig, LlavaNextConfig
from transformers import PretrainedConfig
from typing import List
from transformers import AutoConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig

class LlavaWithResNetConfig(LlavaConfig):
    model_type = "llava_with_resnet"

    def __init__(self, resnet_hidden_size=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resnet_hidden_size = resnet_hidden_size



class LlavaWithVisionExpertConfig(LlavaNextConfig):
    model_type = "llava_with_vision_expert"


    def __init__(self, expert_config=None, resampler_config=None, *args, **kwargs):
        if isinstance(expert_config, dict):
            expert_config["model_type"] = (
                expert_config["model_type"] if "model_type" in expert_config else "resnet_expert"
            )
            expert_config = CONFIG_MAPPING[expert_config["model_type"]](**expert_config)
        self.expert_config = expert_config
        self.resampler_config = resampler_config
        super().__init__(*args, **kwargs)
    
    def to_dict(self):
        output = super().to_dict()
        if self.resampler_config is not None:
            output['resampler_config'] = str(self.resampler_config)
        return output
    
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet_expert"

    def __init__(
        self,
        num_classes: int = 1,
        pretrain_path: str = "",
        use_low_level: str = "npr",
        pretrained: bool = False,
        **kwargs,
    ):

        self.num_classes = num_classes
        self.pretrain_path = pretrain_path
        self.use_low_level = use_low_level
        self.pretrained = pretrained
        super().__init__(**kwargs)




AutoConfig.register("resnet_expert", ResnetConfig)
AutoConfig.register("llava_with_vision_expert", LlavaWithVisionExpertConfig)
