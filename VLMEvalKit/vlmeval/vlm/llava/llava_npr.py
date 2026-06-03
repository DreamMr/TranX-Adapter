from llavanpr.config_llavanpr import LlavaWithVisionExpertConfig, ResnetConfig
from llavanpr.modeling_llavanpr import CustomLlavaNextForConditionalGeneration
from llavanpr.processing_llavanpr import CustomLlavaNextProcessor
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoConfig, AutoModel

import torch
from PIL import Image
from abc import abstractproperty
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests
from io import BytesIO
import numpy as np

def pil_png(img, compress_val=6):
    out = BytesIO()
    #img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    img.save(out, format='PNG', compress_level=0)
    img.load()
    out.close()
    return img

class LLaVA_NextNPR(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="llava-hf/llava-v1.6-vicuna-7b-hf", verbose=False, is_draw=False, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.verbose = verbose
        model = AutoModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.is_draw = is_draw

        model = model.eval()
        self.model = model.cuda()
        self.device = self.model.device
        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=50, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )


    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        img_path = None
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": """Can you determine if this picture is fake or real? Just answer "Real" or "Fake"."""})
            else:
                content.append({"type": "image"})
                img_path = msg['value']
                if isinstance(img_path, str):
                    img = Image.open(msg["value"])
                    img = img.convert("RGB")
                else:
                    img = img_path
                images.append(img)
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(
            self.device, torch.float16
        )
        
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer)
        answer = answer.replace('<unk>', '')
        return answer
