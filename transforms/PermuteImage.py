from itertools import permutations
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class PermuteImage:
    def __init__(self, num_tiles=4):
        self.num_tiles = num_tiles

    def __call__(self, img: Image):
        img_np = np.array(img)
        permuted_img_np = self.permute_image(img_np, self.num_tiles)
        # Convert numpy array back to PIL image
        permuted_img = transforms.functional.to_pil_image(permuted_img_np)
        return permuted_img

    def permute_image(self, img, num_tiles=4):
        tiles_per_row = int(np.sqrt(num_tiles))
        x = int(img.shape[0] // tiles_per_row * tiles_per_row)
        y = int(img.shape[1] // tiles_per_row * tiles_per_row)
        img = img[:x, :y, :]
        height, width, channels = img.shape
        tile_width = int(width // tiles_per_row)
        tile_height = int(height // tiles_per_row)
        tiled_array = img.reshape(tiles_per_row, tile_height, tiles_per_row, tile_width, channels)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiles = tiled_array.reshape(num_tiles, tile_height, tile_width, channels)
        np.random.shuffle(tiles)
        tiled_array = tiles.reshape(tiles_per_row, tiles_per_row, tile_height, tile_width, channels)
        tiled_array = tiled_array.swapaxes(1, 2)
        permuted_img = tiled_array.reshape(height, width, channels)
        if np.array_equal(img, permuted_img):
            permuted_img = self.permute_image(img, num_tiles)
        return permuted_img
    

    @ staticmethod
    def permute_all(img, num_tiles=4):
        permuted_imgs = []
        tiles_per_row = int(np.sqrt(num_tiles))
        x = int(img.shape[0] // tiles_per_row * tiles_per_row)
        y = int(img.shape[1] // tiles_per_row * tiles_per_row)
        img = img[:x, :y, :]
        height, width, channels = img.shape
        tile_width = int(width // tiles_per_row)
        tile_height = int(height // tiles_per_row)
        tiled_array = img.reshape(tiles_per_row, tile_height, tiles_per_row, tile_width, channels)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiles = tiled_array.reshape(num_tiles, tile_height, tile_width, channels)
        tiles_permutations = list(permutations(range(num_tiles), num_tiles))
        tiles_permutations.pop(0)
        for permutation in tiles_permutations:
            premuted_tiles= tiles[permutation, :]
            tiled_array = premuted_tiles.reshape(tiles_per_row, tiles_per_row, tile_height, tile_width, channels)
            tiled_array = tiled_array.swapaxes(1, 2)
            permuted_img = tiled_array.reshape(height, width, channels)
            permuted_imgs.append(permuted_img)
        return permuted_imgs
    
    @ staticmethod
    def create_n_premutations(arr: np.array, n:int ,seed: int):
        np.random.seed(seed)
        permutations_set = set()
        while len(permutations_set) < n:
            perm = tuple(np.random.permutation(arr))
            permutations_set.add(perm)
        return list(permutations_set)

    @ staticmethod
    def permute_by_defined_premutations(img, permutations_set):
        permuted_imgs = []
        num_tiles = len(permutations_set[0])
        tiles_per_row = int(np.sqrt(num_tiles))
        x = int(img.shape[0] // tiles_per_row * tiles_per_row)
        y = int(img.shape[1] // tiles_per_row * tiles_per_row)
        img = img[:x, :y, :]
        height, width, channels = img.shape
        tile_width = int(width // tiles_per_row)
        tile_height = int(height // tiles_per_row)
        tiled_array = img.reshape(tiles_per_row, tile_height, tiles_per_row, tile_width, channels)
        tiled_array = tiled_array.swapaxes(1, 2)
        tiles = tiled_array.reshape(num_tiles, tile_height, tile_width, channels)
        for permutation in permutations_set:
            premuted_tiles= tiles[permutation, :]
            tiled_array = premuted_tiles.reshape(tiles_per_row, tiles_per_row, tile_height, tile_width, channels)
            tiled_array = tiled_array.swapaxes(1, 2)
            permuted_img = tiled_array.reshape(height, width, channels)
            permuted_imgs.append(permuted_img)
        return torch.stack(permuted_imgs)
