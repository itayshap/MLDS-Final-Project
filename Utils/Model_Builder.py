
import torch
from Enums.ModelType import ModelType
from Enums.ModelPath import ModelPath
from modules.HeadlessPretrainedModule import HeadlessPretrainedModule
from torchvision import models
from modules.CustomModule import CustomModule2
from modules.VitModule import VitModule
from modules.BaseModule import BaseModule
from os.path import isfile
from collections import namedtuple
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from Config import TrainParams

from transforms.Transform_Builder import Transform_Builder

class Model_Builder():

    @staticmethod
    def build(modelType: ModelType, device: str, trainParams: TrainParams = None, dataloaders: dict = None, path :ModelPath = ModelPath.BASELINE_CUSTOM.value):
        if modelType==ModelType.PRETRAINED:
            model = HeadlessPretrainedModule(pretrained_model = models.resnet50(pretrained=True), device=device)
            model = model.to(device)
            Model_Builder.load_or_train_model(model, path, dataloaders, trainParams)
            
        elif modelType==ModelType.CUSTOM:
            model = CustomModule2(device=device)
            model = model.to(device)
            model = Model_Builder.load_or_train_model(model, path, dataloaders, trainParams)
        else:
            model = VitModule('nateraw/vit-base-cats-vs-dogs', device)
        return model

    def build_finetuned(model, trainParams: TrainParams = None, dataloaders: dict = None, path :ModelPath = ModelPath.BASELINE_CUSTOM.value):
        model = Model_Builder.load_or_train_model(model, path, dataloaders, trainParams)
        return model
           
    def load_or_train_model(model: BaseModule, path: str, dataloaders: dict, trainParams: TrainParams):
        if isfile(path):
            model.load_state_dict(torch.load(path))
        else:
            model_optimizer = trainParams.optimizer(model.parameters(), trainParams.lr)
            criterion = nn.CrossEntropyLoss()
            model.start_train(criterion, model_optimizer, dataloaders=dataloaders, num_epochs=trainParams.num_epochs, verbose=trainParams.verbose)
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