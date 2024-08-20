
import torch
from Enums.ModelType import ModelType
from modules.HeadlessPretrainedModule import HeadlessPretrainedModule
from torchvision import models
from modules.CustomModule import CustomModule2
from modules.VitModule import VitModule
from os.path import isfile
from collections import namedtuple
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader

from transforms.Transform_Builder import Transform_Builder

namedtuple('TrainParams',['lr', 'num_epochs'])
class Model_Builder():

    @staticmethod
    def build(modelType: ModelType, device: str, trainParams: namedtuple = None, dataloaders: dict = None):
        if modelType==ModelType.PRETRAINED:
            model = HeadlessPretrainedModule(pretrained_model = models.resnet50(pretrained=True), device=device)
            model = model.to(device)
            Model_Builder.load_or_train_model(model, "./pretrained_model.pth", dataloaders, trainParams)
            
        elif modelType==ModelType.CUSTOM:
            model = CustomModule2(device=device)
            model = model.to(device)
            model = Model_Builder.load_or_train_model(model, "./CustomModel.pth", dataloaders, trainParams)
        else:
            model = VitModule('nateraw/vit-base-cats-vs-dogs', device)
        return model

    def load_or_train_model(model: Module, path: str, dataloaders: dict, trainParams: namedtuple):
        if isfile(path):
            model.load_state_dict(torch.load(path))
        else:
            model_optimizer = trainParams.optimizer(model.parameters(), trainParams.lr)
            criterion = nn.CrossEntropyLoss()
            model.start_train(criterion, model_optimizer, dataloaders, num_epochs=trainParams.num_epochs)
            torch.save(model.state_dict(), path)
        return model
    
    @staticmethod
    def build_dataloaders(train, val, test, custom_transform):
        train_set, val_set, test_set = Transform_Builder.transform_datasets((train, val, test), transform=custom_transform)
        dataloaders = {
        'train': DataLoader(train_set, batch_size=32, drop_last=True, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=32, drop_last=True, num_workers=0),
        'test': DataLoader(test_set, batch_size=32, drop_last=True, num_workers=0)}
        return dataloaders