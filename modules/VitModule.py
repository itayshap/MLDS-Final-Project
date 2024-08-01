from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch

class VitModule():
    def __init__(self, model_name_or_path: str, device: torch.device):
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.model = ViTForImageClassification.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.Perdicated_matches = None
        self.test_acc = 0

    def predict_data(self, loader: DataLoader):
        self.Perdicated_matches = torch.empty(0, dtype=torch.bool).to(self.device)
        test_dataset_size = loader.batch_size * len(loader) 
        running_corrects = 0

        with torch.no_grad():
            for batch in loader:
                images, labels = batch['image'], batch['labels']
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.int64)
                output = self.model(images).logits
                y_pred = torch.max(output, dim=1)[1]
                batch_matching = y_pred == labels
                self.Perdicated_matches = torch.concatenate([self.Perdicated_matches, batch_matching])
                running_corrects += batch_matching.sum()
        self.test_acc = round(running_corrects.item() / test_dataset_size, 4)
        print('Test Acc: {:4f}'.format(self.test_acc))