from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

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
        # print('Test Acc: {:4f}'.format(self.test_acc))

    def fine_tune(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, learning_rate: float, verbose=False):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9,0.999), eps=1e-08)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                images, labels = batch['image'], batch['labels']
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.int64)

                optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            self.__validate(val_loader, verbose)

    def __validate(self, val_loader: DataLoader, verbose=False):
        self.model.eval()
        running_corrects = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch['image'], batch['labels']
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.int64)

                outputs = self.model(images).logits
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        val_acc = running_corrects.double() / len(val_loader.dataset)
        if verbose:
            print(f'Validation Acc: {val_acc:.4f}')