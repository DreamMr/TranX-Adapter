import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)

from .base import BaseModel

from .llava import (
    LLaVA,
    LLaVA_Next,
    LLaVA_XTuner,
    LLaVA_Next2,
    LLaVA_OneVision,
    LLaVA_OneVision_HF,
    LLaVA_NextNPR,
)

from .qwen_vl import QwenVL, QwenVLChat
from .qwen2_vl import Qwen2VLChat, Qwen2VLChatAguvis

# AIGC Detection

from .clip import Clip
from .clip_aigi import ClipAIGI
from .qwen3_vl import Qwen3VL_NPR
# try:
#     from .qwen3_vl import Qwen3VL_NPR
# except Exception as e:
#     print("Please check transformers version for Qwen3VL!!!")