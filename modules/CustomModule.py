from modules.BaseModule import BaseModule
from torch import nn
import torch


class CustomModule(BaseModule):
    def __init__(self, device: torch.device, n_classes: int=2):
        super().__init__(device)
        #############################################################################
        # TO DO:                                                                    #
        # Initiate the different layers you wish to use in your network.            #
        # This method has no return value.                                          #
        #############################################################################
        self.layer1 = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        nn.Dropout()
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        nn.Dropout()
        )
        self.layer3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2, 2),
        nn.Dropout()
        )
        self.layer4 = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
       
        )
        self.layer5 = nn.Sequential(
        nn.Linear(512, n_classes)
        )


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TO DO:                                                                    #
        # Define the forward propagation. You need to pass an image through the     #
        # network and obtain class predictions.                                     #
        # This function returns the predication of your model.                      #
        #############################################################################
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

class CustomModule2(BaseModule):
    def __init__(self, device: torch.device, n_classes: int=2):
        super().__init__(device)

        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )  
        self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        nn.Flatten(),
        nn.Dropout()
        )  

        self.layer5 = nn.Sequential(
        nn.LazyLinear(out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=n_classes)
        )      
                
        
        
    def forward(self, x):       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x