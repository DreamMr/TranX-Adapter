import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from vlmeval.smp import *
from .base import BaseModel
from PIL import Image
from ..dataset import DATASET_TYPE, DATASET_MODALITY
import torch
import torchvision.transforms as transforms
import os
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig


class Clip(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, pretrained_model=r'/intern/billwenwang/huggingface/clip-vit-large-patch14', verbose=False, **kwargs):
        print("Start initialize model")
        self.verbose = verbose
        self.pretrained_model = pretrained_model
        RANK = int(os.environ.get('RANK', 0))
        self.device = torch.device(f"cuda:{RANK}" if torch.cuda.is_available() else "cpu")
        print("Start Construct CLIP BackBone")
        self.model = CLIPModel.from_pretrained(pretrained_model)
        self.processor = AutoProcessor.from_pretrained(pretrained_model)
        self.unknown_label = "Can not determine"
        self.real_label = "A photo of a real scene."
        self.fake_label = "A photo of a digitally manipulated."

        self.model.eval()
        self.model.cuda()
        

        print("Finish initialize")

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])

        ## warning: just process single image!!
        return text, images[0]

    def generate_inner(self, message, dataset=None):
        #print("into inference")
        _, image = self.concat_tilist(message)
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        inputs = self.processor(text=[self.real_label, self.fake_label], images=image, return_tensors='pt', padding=True).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        fake_prob = probs[:,1]
        if self.verbose:
            print(f"Prediction prob: {fake_prob}")
        if fake_prob > 0.5:
            return "fake"
        else:
            return "real"

    def forward_features(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features.squeeze()
        return image_features

        
