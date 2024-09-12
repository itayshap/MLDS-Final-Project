from itertools import permutations
import numpy as np
import torch
from torchvision import transforms

class PermuteImage:
    def __init__(self, tiles_num=4, permutations_set=None):
        self.permutations_set = permutations_set
        self.tiles_num = tiles_num 

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.PILToTensor()(img)
        if self.permutations_set == None:
            permuted_img = self.permute_tensor(img, self.tiles_num)
        else: 
            permuted_img = self.permute_by_defined_premutations(img, self.permutations_set)
        return permuted_img
    
    def permute_tensor(self, img: torch.Tensor, tiles_num):
        tiles_per_row = int(np.sqrt(tiles_num))
        x = int(img.shape[1] // tiles_per_row * tiles_per_row)
        y = int(img.shape[2] // tiles_per_row * tiles_per_row)
        img = img[:, :x, :y]
        channels, height, width = img.shape
        tile_width = int(width // tiles_per_row)
        tile_height = int(height // tiles_per_row)
        tiled_array = img.reshape(channels, tiles_per_row, tile_height, tiles_per_row, tile_width)
        tiled_array = tiled_array.swapaxes(2, 3)
        tiles = tiled_array.reshape(channels, tiles_num, tile_height, tile_width)
        idx = torch.randperm(tiles.shape[1])
        tiles = tiles[:,idx]
        tiled_array = tiles.reshape(channels, tiles_per_row, tiles_per_row, tile_height, tile_width)
        tiled_array = tiled_array.swapaxes(2, 3)
        permuted_img = tiled_array.reshape(channels, height, width)
        if np.array_equal(img, permuted_img):
            permuted_img = self.permute_tensor(img, tiles_num)
        return permuted_img 

    def permute_by_defined_premutations(self, img: torch.Tensor, permutations_set):
        permuted_imgs = []
        tiles_per_row = int(np.sqrt(self.tiles_num))
        x = int(img.shape[1] // tiles_per_row * tiles_per_row)
        y = int(img.shape[2] // tiles_per_row * tiles_per_row)
        img = img[:, :x, :y]
        channels, height, width = img.shape
        tile_width = int(width // tiles_per_row)
        tile_height = int(height // tiles_per_row)
        tiled_array = img.reshape(channels, tiles_per_row, tile_height, tiles_per_row, tile_width)
        tiled_array = tiled_array.swapaxes(2, 3)
        tiles = tiled_array.reshape(channels, self.tiles_num, tile_height, tile_width)
        for permutation in permutations_set:
            premuted_tiles= tiles[:, permutation]
            tiled_array = premuted_tiles.reshape(channels, tiles_per_row, tiles_per_row, tile_height, tile_width)
            tiled_array = tiled_array.swapaxes(2, 3)
            permuted_img = tiled_array.reshape(channels, height, width)
            permuted_imgs.append(permuted_img)
        return torch.stack(permuted_imgs).squeeze()