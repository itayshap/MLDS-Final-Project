
import pandas as pd
from transforms.Transform_Builder import Transform_Builder
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from modules.BaseModule import BaseModule
from Utils.Utilities import calculate_tiles_derivative
import numpy as np

def run_tile_num_test(models_params: dict, tiles_num_options: list, test: Dataset):
    results_df = pd.DataFrame([])
    for model_name, params in models_params.items():
        results = {"Tiles": [], "Accuracy": []}
        for tiles_num in tiles_num_options:
                transform_with_premute = Transform_Builder.build(size=params['size'], with_premute=True, tiles_num=tiles_num, is_vit=params['is_vit'], processor=params['processor'])
                premuted_test_set = test.with_transform(transform_with_premute)
                premuted_dataloader = DataLoader(premuted_test_set, batch_size=32, drop_last=True, num_workers=0)
                params['Model'].predict_data(premuted_dataloader)
                results["Tiles"].append(tiles_num)
                results["Accuracy"].append(params['Model'].test_acc)
        df = pd.DataFrame(results)
        df["Model"] = model_name
        results_df = pd.concat([results_df, df])
    return results_df

def run_permutation_rank_test(model : BaseModule, test: Dataset, permutation_dict: dict, base_score, is_vit=False, processor = None, size:int =224):
        results = {'Score': [], 'Accuracy': []}
        for premutation in permutation_dict.values():
                permuted_test_set = test.with_transform(Transform_Builder.build(size=size, with_premute=True, tiles_num=len(premutation), permutations_set= [premutation], is_vit=is_vit, processor=processor))
                dataloader = DataLoader(permuted_test_set, batch_size=32, drop_last=True, num_workers=0)
                results['Score'].append(np.abs(calculate_tiles_derivative(permuted_test_set[:]['image'])-base_score)/len(permuted_test_set))
                model.predict_data(dataloader)
                results['Accuracy'].append(model.test_acc)
        return results

def run_accuracy_test(model : BaseModule, test: Dataset, permutation_dict: dict, is_vit=False, processor = None, size:int =224):
        results = {'Permutation': [], 'Accuracy': []}
        for premutation_name, premutation in permutation_dict.items():
                permuted_test_set = test.with_transform(Transform_Builder.build(size=size, with_premute=True, tiles_num=len(premutation), permutations_set= [premutation], is_vit=is_vit, processor=processor))
                dataloader = DataLoader(permuted_test_set, batch_size=32, drop_last=True, num_workers=0)
                results['Permutation'].append(premutation_name)
                model.predict_data(dataloader)
                results['Accuracy'].append(model.test_acc)
        return results