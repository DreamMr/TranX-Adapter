from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import torch
model_name = r'openai/clip-vit-large-patch14-336'
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(["A photo of a digitally manipulated."], padding=True, return_tensors="pt")
with torch.inference_mode():
    text_features = model.get_text_features(**inputs)
    print(text_features.shape)
save_path= './text_embedding.pth'
torch.save({
    "features": text_features
}, save_path)