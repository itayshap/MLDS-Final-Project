import torch
from torchvision import transforms
from transforms.PermuteImage import PermuteImage

class Transform_Builder:
    
    @staticmethod
    def build(size : int = 224, with_premute : bool = False, num_tiles: int = 4):
        model_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if with_premute:
            model_transform.transforms.insert(1, PermuteImage(num_tiles=num_tiles))
        
        def transform(examples: dict):
            lst = []
            if examples.get('image'):
                for img in examples['image']:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    lst.append(model_transform(img))
                examples['image'] = lst
            examples['labels'] = torch.tensor(examples['labels'], dtype=torch.int64)
            return examples
        return transform