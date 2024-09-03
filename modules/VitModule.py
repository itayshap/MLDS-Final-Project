from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from modules.BaseModule import BaseModule

class VitModule(BaseModule):
    def __init__(self, model_name_or_path: str, device: torch.device):
        super().__init__(device)
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.model = ViTForImageClassification.from_pretrained(model_name_or_path).to(device)

        for name,p in self.model.named_parameters():
            if not name.startswith('classifier'):
                p.requires_grad = False

    def forward(self, images):
        return self.model(images).logits
