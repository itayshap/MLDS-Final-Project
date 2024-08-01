from collections import defaultdict
from time import time
from typing import Dict
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import copy

class BaseModule(Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.test_acc = 0
        self.Perdicated_matches = None

    def start_train(self, criterion, optimizer: torch.optim.Optimizer, dataloaders: Dict[str, DataLoader], num_epochs=5, seed=42):
        dataset_sizes = {k: v.batch_size *
                         len(v) for k, v in dataloaders.items()}
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        since = time()
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.state_dict())

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                    torch.set_grad_enabled(True)
                else:
                    self.eval()
                    torch.set_grad_enabled(False)

                running_loss = 0.0   # total loss of the network at each epoch
                running_corrects = 0  # number of correct predictions

                # Iterate over data.
                for batch in dataloaders[phase]:

                    images, labels = batch['image'], batch['labels']
                    images = images.to(self.device)
                    labels = labels.to(self.device, dtype=torch.int64)
                    optimizer.zero_grad()
                    output = self(images)
                    loss = criterion(output, labels)

                    running_loss += loss.item()
                    y_pred = torch.max(output, dim=1)[1]
                    running_corrects += (y_pred == labels).sum()

                    if self.training:
                        loss.backward()
                        optimizer.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.item() / dataset_sizes[phase]

                print('{} Loss: {:.4f}  |  Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.state_dict())

                history = self.train_history if self.training else self.val_history
                history['loss'].append(epoch_loss)
                history['accuracy'].append(epoch_acc)

        time_elapsed = time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        return self.load_state_dict(best_model_wts)
    
    def predict_data(self, loader: DataLoader):
        self.Perdicated_matches = torch.empty(0, dtype=torch.bool).to(self.device)
        test_dataset_size = loader.batch_size * len(loader) 
        running_corrects = 0
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                images, labels = batch['image'], batch['labels']
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.int64)

                output = self(images)
                y_pred = torch.max(output, dim=1)[1]
                batch_matching = y_pred == labels
                self.Perdicated_matches = torch.concatenate([self.Perdicated_matches, batch_matching])
                running_corrects += batch_matching.sum()
        self.test_acc = running_corrects.item() / test_dataset_size
        print('Test Acc: {:4f}'.format(self.test_acc))
        