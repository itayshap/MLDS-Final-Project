from modules.BaseModule import BaseModule
from torch import nn
import torch


class HeadlessPretrainedModule(BaseModule):
    def __init__(self, pretrained_model: nn.Module, device: torch.device, n_classes: int=2):
        super().__init__(device)
        self.linear = None

        for param in pretrained_model.parameters():
            param.requires_grad_(False)

        modules = list(pretrained_model.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.linear = nn.Linear(pretrained_model.fc.in_features, n_classes)

    def forward(self, images):

        features = self.model(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        
        return features
