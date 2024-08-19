import torch
from torchvision import transforms
from transforms.PermuteImage import PermuteImage
from modules.VitModule import ViTImageProcessor
from torch.utils.data import Dataset

def convert_to_rgb(example_batch, transformer = None):
    lst = []
    if example_batch.get('image'):
        for img in example_batch['image']:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if transformer !=None:
                img = transformer(img)
            lst.append(img)
    return lst

class Transform_Builder:
    
    @staticmethod
    def build(size : int = 224, with_premute : bool = False, tiles_num: int = 4, permutations_set = None, is_vit: bool = False, processor = None):
        if is_vit:
            transform = Transform_Builder.build_vit_transform(processor, with_premute, tiles_num, permutations_set)
        else:
            transform = Transform_Builder.bulid_transform(size, with_premute, tiles_num, permutations_set)

        return transform
    
    def bulid_transform(size : int = 224, with_premute : bool = False, tiles_num: int = 4, permutations_set = None):         
        model_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if with_premute:
            model_transform.transforms.insert(3, PermuteImage(tiles_num=tiles_num, permutations_set=permutations_set))
        
        def transform(examples: dict):
            lst = convert_to_rgb(examples, model_transform)
            examples['image'] = torch.stack(lst)
            examples['labels'] = torch.tensor(examples['labels'], dtype=torch.int64)
            return examples
        return transform
    
    def build_vit_transform(processor: ViTImageProcessor, with_premute : bool = False, tiles_num: int = 4, permutations_set = None):
        permuter = PermuteImage(tiles_num, permutations_set) if with_premute else None
        def vit_transform(example_batch):
            lst = convert_to_rgb(example_batch, permuter)
            example_batch['image'] = processor(lst, return_tensors='pt')['pixel_values']
            example_batch['labels'] = torch.tensor(example_batch['labels'], dtype=torch.int64) 
            return example_batch
        return vit_transform  
    
    @staticmethod
    def transform_datasets(datasets: tuple[Dataset], transform = callable):
        return (dataset.with_transform(transform) for dataset in datasets) 
