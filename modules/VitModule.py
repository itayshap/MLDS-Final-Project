from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch
from datasets import Dataset
from transforms.PermuteImage import PermuteImage

class VitModule():
    def __init__(self, model_name_or_path: str, device: torch.device, n_classes: int=2):
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.model = ViTForImageClassification.from_pretrained(model_name_or_path).to(device)
        self.device = device

    def predict_data(self, loader: DataLoader):
        total_matching = torch.empty(0, dtype=torch.bool).to(self.device)
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
                total_matching = torch.concatenate([total_matching, batch_matching])
                running_corrects += batch_matching.sum()
        print('Test Acc: {:4f}'.format(running_corrects.item() / test_dataset_size))
        return total_matching
    
    def _transform(self, example_batch: dict, with_permute: bool = False):
        # Take a list of PIL images and turn them to pixel values
        img = example_batch['image']
        if with_permute:
            img = PermuteImage()(img)
        example_batch['image'] = self.processor(img, return_tensors='pt')['pixel_values']
        return example_batch
    
    def transform(self, examples):
        if examples.get('image'):
            for img in examples['image']:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self._transform(img)
                examples['image'] = img
            examples['labels'] = torch.tensor(examples['labels'], dtype=torch.int64)
        return examples 
    
    def tranform_data(self, dataset : Dataset):
        return dataset.with_transform()
