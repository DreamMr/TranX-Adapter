import time
load_vlm_time = time.time()
from vlmeval.vlm import *
print(f"Loading VLM module cost time: {(time.time() - load_vlm_time) / 60.}")

load_api_time = time.time()
from vlmeval.api import *
print(f"Loading API module cost time: {(time.time() - load_api_time) / 60.}")
from functools import partial
import os


PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = "Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. "



api_cost_time = time.time()
o1_key = os.environ.get('O1_API_KEY', None)
o1_base = os.environ.get('O1_API_BASE', None)
o1_apis = {
    'o1': partial(
        GPT4V,
        model="o1-2024-12-17",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800, 
        max_tokens=16384,
        verbose=False,

    ),
    'o3': partial(
        GPT4V, 
        model="o3-2025-04-16",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800, 
        max_tokens=16384, 
        verbose=False,
    ),
    'o4-mini': partial(
        GPT4V, 
        model="o4-mini-2025-04-16",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
}

api_models = {
    # GPT
    "GPT4V": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_HIGH": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409_HIGH": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4o_HIGH": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20240806": partial(
        GPT4V,
        model="gpt-4o-2024-08-06",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20241120": partial(
        GPT4V,
        model="gpt-4o-2024-11-20",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "ChatGPT4o": partial(
        GPT4V,
        model="chatgpt-4o-latest",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_MINI": partial(
        GPT4V,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4.5": partial(
        GPT4V, 
        model='gpt-4.5-preview-2025-02-27',
        temperature=0, 
        timeout=600,
        img_size=-1, 
        img_detail='high', 
        retry=10, 
        verbose=False,
    ),
    "gpt-4.1-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-mini-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-mini-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-nano-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    # Gemini
    "GeminiPro1-0": partial(
        Gemini, model="gemini-1.0-pro", temperature=0, retry=10
    ),  # now GeminiPro1-0 is only supported by vertex backend
    "GeminiPro1-5": partial(
        Gemini, model="gemini-1.5-pro", temperature=0, retry=10
    ),
    "GeminiFlash1-5": partial(
        Gemini, model="gemini-1.5-flash", temperature=0, retry=10
    ),
    "GeminiPro1-5-002": partial(
        GPT4V, model="gemini-1.5-pro-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash1-5-002": partial(
        GPT4V, model="gemini-1.5-flash-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash2-0": partial(
        Gemini, model="gemini-2.0-flash", temperature=0, retry=10
    ),
    "GeminiFlashLite2-0": partial(
        Gemini, model="gemini-2.0-flash-lite", temperature=0, retry=10
    ),
    "GeminiFlash2-5": partial(
        GPT4V, model="gemini-2.5-flash", temperature=0, retry=10, timeout=1800
    ),
    "GeminiPro2-5": partial(
        GPT4V, model="gemini-2.5-pro", temperature=0, retry=10, timeout=1800
    ),
    
    # Qwen-VL
    "QwenVLPlus": partial(QwenVLAPI, model="qwen-vl-plus", temperature=0, retry=10),
    "QwenVLMax": partial(QwenVLAPI, model="qwen-vl-max", temperature=0, retry=10),
    "QwenVLMax-250408": partial(QwenVLAPI, model="qwen-vl-max-2025-04-08", temperature=0, retry=10),

    # Reka
    "RekaEdge": partial(Reka, model="reka-edge-20240208"),
    "RekaFlash": partial(Reka, model="reka-flash-20240226"),
    "RekaCore": partial(Reka, model="reka-core-20240415"),
    # Step1V
    "Step1V": partial(
        GPT4V,
        model="step-1v-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1.5V-mini": partial(
        GPT4V,
        model="step-1.5v-mini",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1o": partial(
        GPT4V,
        model="step-1o-vision-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    # Yi-Vision
    "Yi-Vision": partial(
        GPT4V,
        model="yi-vision",
        api_base="https://api.lingyiwanwu.com/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # Claude
    "Claude3V_Opus": partial(
        Claude3V, model="claude-3-opus-20240229", temperature=0, retry=10, verbose=False
    ),
    "Claude3V_Sonnet": partial(
        Claude3V,
        model="claude-3-sonnet-20240229",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3V_Haiku": partial(
        Claude3V,
        model="claude-3-haiku-20240307",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet": partial(
        Claude3V,
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet_20241022": partial(
        Claude3V,
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-7V_Sonnet": partial(
        Claude3V,
        model="claude-3-7-sonnet-20250219",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude4_Opus": partial(
        Claude3V,
        model="claude-4-opus-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    "Claude4_Sonnet": partial(
        Claude3V,
        model="claude-4-sonnet-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    # GLM4V
    "GLM4V": partial(GLMVisionAPI, model="glm4v-biz-eval", temperature=0, retry=10),
    "GLM4V_PLUS": partial(GLMVisionAPI, model="glm-4v-plus", temperature=0, retry=10),
    "GLM4V_PLUS_20250111": partial(
        GLMVisionAPI, model="glm-4v-plus-0111", temperature=0, retry=10
    ),
    # MiniMax abab
    "abab6.5s": partial(
        GPT4V,
        model="abab6.5s-chat",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "abab7-preview": partial(
        GPT4V,
        model="abab7-chat-preview",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # CongRong
    "CongRong-v1.5": partial(CWWrapper, model="cw-congrong-v1.5", temperature=0, retry=10),
    "CongRong-v2.0": partial(CWWrapper, model="cw-congrong-v2.0", temperature=0, retry=10),
    # SenseNova
    "SenseNova-V6-Pro": partial(
        SenseChatVisionAPI, model="SenseNova-V6-Pro", temperature=0, retry=10
    ),
    "SenseNova-V6-Reasoner": partial(
        SenseChatVisionAPI, model="SenseNova-V6-Reasoner", temperature=0, retry=10
    ),
    "HunYuan-Vision": partial(
        HunyuanVision, model="hunyuan-vision", temperature=0, retry=10
    ),
    "HunYuan-Standard-Vision": partial(
        HunyuanVision, model="hunyuan-standard-vision", temperature=0, retry=10
    ),
    "HunYuan-Large-Vision": partial(
        HunyuanVision, model="hunyuan-large-vision", temperature=0, retry=10
    ),
    "BailingMM-Lite-1203": partial(
        bailingMMAPI, model="BailingMM-Lite-1203", temperature=0, retry=10
    ),
    "BailingMM-Pro-0120": partial(
        bailingMMAPI, model="BailingMM-Pro-0120", temperature=0, retry=10
    ),
    # BlueLM-V
    "BlueLM_V": partial(BlueLM_V_API, model="BlueLM-VL-v3.0", temperature=0, retry=10),
    # JiuTian-VL
    "JTVL": partial(JTVLChatAPI, model="jt-vl-chat", temperature=0, retry=10),
    "Taiyi": partial(TaiyiAPI, model="taiyi", temperature=0, retry=10),
    # TeleMM
    "TeleMM": partial(TeleMMAPI, model="TeleAI/TeleMM", temperature=0, retry=10),
    "Qwen2.5-VL-32B-Instruct-SiliconFlow": partial(
        SiliconFlowAPI, model="Qwen/Qwen2.5-VL-32B-Instruct", temperature=0, retry=10),
    # lmdeploy api
    "lmdeploy": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "lmdeploy_internvl_78B_MPO": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=100,
    ),
    "lmdeploy_qvq_72B_preview": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'Taichu-VLR-3B': partial(
        TaichuVLRAPI, 
        model='taichu_vlr_3b', 
        url="https://platform.wair.ac.cn/maas/v1/chat/completions"
    ),
    'Taichu-VLR-7B': partial(
        TaichuVLRAPI, 
        model='taichu_vlr_7b', 
        url="https://platform.wair.ac.cn/maas/v1/chat/completions"
    ),
    # doubao_vl
    "DoubaoVL": partial(
        DoubaoVL, model="Doubao-1.5-vision-pro", temperature=0, retry=10, verbose=False
    ),
    "Seed1.5-VL": partial(
        DoubaoVL, 
        model="doubao-1-5-thinking-vision-pro-250428", 
        temperature=0,
        retry=10, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6": partial(
        DoubaoVL, 
        model="doubao-seed-1.6-250615", 
        temperature=0,
        retry=10, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6-Flash": partial(
        DoubaoVL, 
        model="doubao-seed-1.6-flash-250615", 
        temperature=0,
        retry=10, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6-Thinking": partial(
        DoubaoVL, 
        model="doubao-seed-1.6-thinking-250615", 
        temperature=0,
        retry=10, 
        verbose=False, 
        max_tokens=16384,
    ),
    # Shopee MUG-U
    'MUG-U-7B': partial(
        MUGUAPI, 
        model='MUG-U', 
        temperature=0,  
        retry=10, 
        verbose=False, 
        timeout=300),
    # grok
    "grok-vision-beta": partial(
        GPT4V,
        model="grok-vision-beta",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-2-vision-1212": partial(
        GPT4V,
        model="grok-2-vision",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # kimi
    "moonshot-v1-8k": partial(
        GPT4V,
        model="moonshot-v1-8k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-32k": partial(
        GPT4V,
        model="moonshot-v1-32k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-128k": partial(
        GPT4V,
        model="moonshot-v1-128k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "vllm_model_llava_next_mistral": partial(
        VLLMWrapper,
        model="llava_next_mistral",
        api_base="http://localhost:23335/v1"
    ),
    "vllm_model_qwen2_5vl": partial(
        VLLMWrapper,
        model="qwen2_5",
        api_base="http://localhost:23335/v1"
    )
}

print("API cost time: {} mins.".format((time.time() - api_cost_time) / 60.))

xtuner_series = {
    "llava-internlm2-7b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-7b",
        llava_path="xtuner/llava-internlm2-7b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm2-20b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-20b",
        llava_path="xtuner/llava-internlm2-20b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm-7b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm-chat-7b",
        llava_path="xtuner/llava-internlm-7b",
        visual_select_layer=-2,
        prompt_template="internlm_chat",
    ),
    "llava-v1.5-7b-xtuner": partial(
        LLaVA_XTuner,
        llm_path="lmsys/vicuna-7b-v1.5",
        llava_path="xtuner/llava-v1.5-7b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-v1.5-13b-xtuner": partial(
        LLaVA_XTuner,
        llm_path="lmsys/vicuna-13b-v1.5",
        llava_path="xtuner/llava-v1.5-13b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-llama-3-8b": partial(
        LLaVA_XTuner,
        llm_path="xtuner/llava-llama-3-8b-v1_1",
        llava_path="xtuner/llava-llama-3-8b-v1_1",
        visual_select_layer=-2,
        prompt_template="llama3_chat",
    ),
}

qwen_series = {
    "qwen_base": partial(QwenVL, model_path="Qwen/Qwen-VL"),
    "qwen_chat": partial(QwenVLChat, model_path="Qwen/Qwen-VL-Chat"),
    # "monkey": partial(Monkey, model_path="echo840/Monkey"),
    # "monkey-chat": partial(MonkeyChat, model_path="echo840/Monkey-Chat"),
    # "minimonkey": partial(MiniMonkey, model_path="mx262/MiniMonkey"),
}

llava_series = {
    "llava_v1.5_7b": partial(LLaVA, model_path="liuhaotian/llava-v1.5-7b"),
    "llava_v1.5_13b": partial(LLaVA, model_path="liuhaotian/llava-v1.5-13b"),
    "llava_v1_7b": partial(LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),
    "sharegpt4v_7b": partial(LLaVA, model_path="Lin-Chen/ShareGPT4V-7B"),
    "sharegpt4v_13b": partial(LLaVA, model_path="Lin-Chen/ShareGPT4V-13B"),
    "llava_next_vicuna_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-7b-hf"
    ),
    "llava_next_vicuna_13b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-13b-hf"
    ),
    "llava_next_mistral_7b": partial(
        LLaVA_Next, model_path="/home/workdir/LVLM/huggingface/llava-v1.6-mistral-7b-hf"
    ),
    "llava_next_yi_34b": partial(LLaVA_Next, model_path="llava-hf/llava-v1.6-34b-hf"),
    "llava_next_llama3": partial(
        LLaVA_Next, model_path="llava-hf/llama3-llava-next-8b-hf"
    ),
    "llava_next_72b": partial(LLaVA_Next, model_path="llava-hf/llava-next-72b-hf"),
    "llava_next_110b": partial(LLaVA_Next, model_path="llava-hf/llava-next-110b-hf"),
    "llava_next_qwen_32b": partial(
        LLaVA_Next2, model_path="lmms-lab/llava-next-qwen-32b"
    ),
    "llava_next_interleave_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-hf"
    ),
    "llava_next_interleave_7b_dpo": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-dpo-hf"
    ),
    "llava-onevision-qwen2-0.5b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ),
    "llava-onevision-qwen2-0.5b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    ),
    "llava-onevision-qwen2-7b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    ),
    "llava-onevision-qwen2-7b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-si-hf"
    ),
    "llava_onevision_qwen2_0.5b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-si"
    ),
    "llava_onevision_qwen2_7b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-si"
    ),
    "llava_onevision_qwen2_72b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-si"
    ),
    "llava_onevision_qwen2_0.5b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-ov"
    ),
    "llava_onevision_qwen2_7b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-ov"
    ),
    "llava_onevision_qwen2_72b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-ov-sft"
    ),
    "Aquila-VL-2B": partial(LLaVA_OneVision, model_path="BAAI/Aquila-VL-2B-llava-qwen"),
    "llava_video_qwen2_7b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
    ),
    "llava_video_qwen2_72b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-72B-Qwen2"
    ),
    "varco-vision-hf": partial(
        LLaVA_OneVision_HF, model_path="NCSOFT/VARCO-VISION-14B-HF"
    ),
}




qwen2vl_series = {
    "Qwen-VL-Max-0809": partial(
        Qwen2VLAPI,
        model="qwen-vl-max-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen-VL-Plus-0809": partial(
        Qwen2VLAPI,
        model="qwen-vl-plus-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "QVQ-72B-Preview": partial(
        Qwen2VLChat,
        model_path="Qwen/QVQ-72B-Preview",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        system_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
        max_new_tokens=8192,
        post_process=False,
    ),
    "Qwen2-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "XinYuan-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Cylingo/Xinyuan-VL-2B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2.5-VL-3B-Instruct": partial(
        Qwen2VLChat,
        model_path="/home/workdir/LVLM/huggingface/Qwen2.5-VL-3B-Instruct",
        min_pixels=512*512,
        max_pixels=512*512,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-3B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-32B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MiMo-VL-7B-SFT": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "MiMo-VL-7B-RL": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-RL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "Qwen2.5-VL-72B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-Omni-7B-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_audio_in_video=True, # set use audio in video
    ),
    "Qwen2.5-Omni-7B": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
}



aigc_detection = {
    "LLaVA-Next_expert_projector_sft_step": partial(LLaVA_NextNPR, model_path='/intern/billwenwang/checkpoints/llava_v16_sft_w_expert_projector_sft/v0-20250804-121433/checkpoint-1016-merged'),
    "Qwen3VL_NPR": Qwen3VL_NPR
}


supported_VLM = {}

model_groups = [
    o1_apis, api_models, qwen_series, llava_series, 
    qwen2vl_series, aigc_detection, 
]

for grp in model_groups:
    supported_VLM.update(grp)
