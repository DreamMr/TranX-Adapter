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
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig, CLIPVisionModel


class CLIP_Model(nn.Module):
    def __init__(self, path):
        super(CLIP_Model, self).__init__()

        self.path = path
        self.model = CLIPVisionModel.from_pretrained(path)

        # add proj
        proj_state_dict = torch.load(os.path.join(self.path,'proj.pth'))
        self.proj = nn.Parameter(torch.randn(proj_state_dict.shape))
        self.proj.data.copy_(proj_state_dict)

        # add classifier head
        fc_state_dict = torch.load(os.path.join(self.path, 'fc.pth'))
        self.fc = nn.Linear(fc_state_dict['weight'].shape[1], fc_state_dict['weight'].shape[0])
        self.fc.load_state_dict(fc_state_dict)
    
    def forward(self, inputs):

        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        feature = pooled_output @ self.proj
        logits = self.fc(feature)
        return logits


class ClipAIGI(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, pretrained_model=r'/intern/billwenwang/huggingface/clip-vit-large-patch14', verbose=False, **kwargs):
        #print("Start initialize model")
        self.verbose = verbose
        self.pretrained_model = pretrained_model
        RANK = int(os.environ.get('RANK', 0))
        self.device = torch.device(f"cuda:{RANK}" if torch.cuda.is_available() else "cpu")
        #print("Start Construct CLIP BackBone")
        self.model = CLIP_Model(pretrained_model)
        self.processor = AutoProcessor.from_pretrained(pretrained_model)
        
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

        inputs = self.processor(images=image, return_tensors='pt', padding=True).to(self.device)
        with torch.inference_mode():
            logits = self.model(inputs)
            fake_probs = logits.sigmoid().flatten().tolist()[0]
        
        if self.verbose:
            print(f"Prediction prob: {fake_probs}")
        if fake_probs > 0.5:
            return "fake"
        else:
            return "real"

        
