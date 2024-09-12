
import torch
from Enums.ModelType import ModelType
from Enums.ModelPath import ModelPath
from modules.HeadlessPretrainedModule import HeadlessPretrainedModule
from torchvision import models
from modules.CustomModule import CustomBetterModule, CustomModule
from modules.VitModule import VitModule
from modules.BaseModule import BaseModule
from os.path import isfile
from torch import nn
from torch.utils.data import DataLoader
from Config import TrainParams

from transforms.Transform_Builder import Transform_Builder

class Model_Builder():

    @staticmethod
    def build(modelType: ModelType, device: str, trainParams: TrainParams = None, dataloaders: dict = None, path :ModelPath = None):
        if modelType==ModelType.PRETRAINED:
            model = HeadlessPretrainedModule(pretrained_model = models.resnet50(pretrained=True), device=device)
            model = model.to(device)      
        elif modelType==ModelType.CUSTOM:
            model = CustomModule(device=device)
            model = model.to(device)
        elif modelType==ModelType.IMPROVED_CUSTOM:
            model = CustomBetterModule(device=device)
            model = model.to(device)
        else:
            model = VitModule('nateraw/vit-base-cats-vs-dogs', device)

        if path != None:
            model = Model_Builder.load_or_train_model(modelType, model, path, dataloaders, trainParams)

        return model
           
    def load_or_train_model(modelType: ModelType, model: BaseModule, path: str, dataloaders: dict, trainParams: TrainParams):
        try:
            if isfile(path):
                if modelType == ModelType.VIT:
                    model.model.classifier.load_state_dict(torch.load(path))
                else: 
                    model.load_state_dict(torch.load(path))
            else:
                model_optimizer = trainParams.optimizer(model.parameters(), **trainParams.optimizer_params)
                criterion = nn.CrossEntropyLoss()
                model.start_train(criterion, model_optimizer, dataloaders=dataloaders, num_epochs=trainParams.num_epochs, verbose=trainParams.verbose)
                if modelType == ModelType.VIT:
                    torch.save(model.model.classifier.state_dict(), path)
                else:
                    torch.save(model.state_dict(), path)
        finally:
            return model
    
    @staticmethod
    def build_dataloaders(train, val, test, custom_transform):
        train_set, val_set, test_set = Transform_Builder.transform_datasets((train, val, test), transform=custom_transform)
        dataloaders = {
        'train': DataLoader(train_set, batch_size=32, drop_last=True, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=32, drop_last=True, num_workers=0),
        'test': DataLoader(test_set, batch_size=32, drop_last=True, num_workers=0)}
        return dataloaders