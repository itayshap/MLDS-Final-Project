import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def denormalize(tensor: torch.tensor, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


### draw functions:
def draw_bar_plot(data: pd.DataFrame, x: str,y: str, title: str, figsize=(6,3), ylim=None, hue=None, legend_loc= "lower right", value_verbose = True, palette='pastel', rotation=0):
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.barplot(data = data, y=y, x=x, palette=palette, hue=hue , ax=ax)
    if value_verbose:
        for i in g.containers:
            g.bar_label(i, padding=-20, fmt='%.3f')
    g.set_title(title,  fontdict={'weight': 'bold'})
    g.set_ylim(ylim)
    g.set_xlabel(g.get_xlabel(), fontdict={'weight': 'bold'})
    g.set_ylabel(g.get_ylabel(), fontdict={'weight': 'bold'})
    if hue:
        sns.move_legend(g, legend_loc)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show();

def draw_images(images: torch.Tensor, titles: dict, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
    to_delete = False
    num_images = images.shape[0]
    if num_images % 2 == 0:
        fig, axes = plt.subplots(2, int(num_images/2), figsize=(9,5))
    elif np.sqrt(num_images) ** 2 == num_images:
        fig, axes = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)), figsize= (8,8))
    else:
         fig, axes = plt.subplots(2, int(num_images/2+1), figsize=(9,5))
         to_delete = True
    axes = axes.flatten()
    for i, (key, value) in enumerate(titles.items()):
        img = torch.permute(denormalize(images[i,:],mean, std),(1,2,0))
        axes[i].imshow(img)
        axes[i].set_title(f'{key} - \n {value}')
        axes[i].axis('off')
    if to_delete:
        fig.delaxes(axes[-1])
    plt.tight_layout(pad=2)
    plt.show();

def draw_rank_test_results(results, permutation_dict, model_name):
    df= pd.DataFrame(results)
    df['Permutation'] = permutation_dict.keys()
    df['Permutations Rank'] = df.Score.rank() -1
    sns.lineplot(data= df, y='Accuracy', x='Permutations Rank')
    for i, row in df.iterrows():
        plt.scatter( y=row.Accuracy, x=row['Permutations Rank'], label=row.Permutation)
    plt.legend()
    plt.title(f'{model_name} - Accuracy Vs Permutation Rank')
    plt.show();


def calculate_tiles_derivative(img: torch.Tensor, tiles_num : int =9) -> float:
    tiles_per_row = int(np.sqrt(tiles_num))
    x = int(img.shape[-2] // tiles_per_row * tiles_per_row)
    y = int(img.shape[-1] // tiles_per_row * tiles_per_row)
    img = img[...,:x, :y]
    tiles_edges_idx = np.linspace(0,img.shape[-2], tiles_per_row + 1, dtype=int)[1:-1]
    gradient = torch.abs((img[...,tiles_edges_idx] - img[...,tiles_edges_idx-1])).sum().item()
    gradient += torch.abs((img[...,tiles_edges_idx,:] - img[...,tiles_edges_idx-1,:])).sum().item()
    return gradient

def create_premutation(tiles_num: int) -> dict:
    permutation_dict = {}
    tiles_per_row = int(np.sqrt(tiles_num))
    tiles_arr = np.arange(tiles_num).reshape(tiles_per_row,tiles_per_row)
    permutation_dict["Original"]  = list(np.arange(tiles_num))
    permutation_dict["First col"]  = list(np.hstack([tiles_arr[:,0][::-1].reshape(-1,1), tiles_arr[:,1:]]).flatten())
    permutation_dict["Last col"]  = list(np.hstack([tiles_arr[:,:-1],tiles_arr[:,-1][::-1].reshape(-1,1)]).flatten())
    permutation_dict["Cols swap"] = list(tiles_arr[:,np.arange(tiles_per_row)[::-1]].flatten())
    horizontal_swap = tiles_arr[np.arange(tiles_per_row)[::-1],:]
    permutation_dict["Rows swap"] = list(horizontal_swap.flatten())
    # permutation_dict["Middle cols"] = list(np.hstack([tiles_arr[:,0].reshape(-1,1),tiles_arr[:,1:-1][::-1],tiles_arr[:,-1].reshape(-1,1)]).flatten())
    permutation_dict["Double swap"] = list(np.arange(tiles_num)[::-1])
    permutation_dict["90 degree flip"] = list(tiles_arr.T[:,np.arange(tiles_per_row)[::-1]].flatten())
    return permutation_dict



