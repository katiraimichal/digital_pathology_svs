import os
import numpy as np
import openslide
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

matplotlib.use('TkAgg')


class DPDataset(Dataset):
    """
    Digital Pathology Dataset
    """

    def __init__(self, files_dir, csv_path, train=True, preprocess=None, transform=None):
        """
        :param files_dir: SVS files directory
        :param csv_path: CSV file directory
        :param train: set as True in case of train mode. In case of validation/test mode set as False
        :param preprocess: Preprocessing to be applied on the data
        :param transform: Transformations to be applied on the data
        """
        self.files_dir = files_dir
        self.preprocess = preprocess
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        svs_dir = os.path.join(self.files_dir, self.df['file_name'][idx])
        patches = self.extract_tiles(svs_dir)
        label = self.df['tag'][idx]

        if self.preprocess is not None:
            patches = [self.preprocess(p) for p in patches]

        if self.train:
            if self.transform is not None:
                patches = [self.transform(p) for p in patches]

        patches = torch.stack(patches, dim=0)

        return patches, label

    def extract_tiles(self, svs_file_path, tile_size=512, white_threshold=0.9, std_threshold=0.04):
        """
        Given a SVS file, return its tiles
        :param svs_file_path:
        :param tile_size:
        :param white_threshold: The proportion of white pixels required to classify the tile as background
        :param std_threshold: Standard deviation threshold for classifying tiles as background based on variation in pixel values
        :return:
        """
        slide = openslide.OpenSlide(svs_file_path)

        tiles = []

        for l in range(0, slide.level_count):
            w, h = slide.level_dimensions[l]

            for x in range(0, w, tile_size):
                for y in range(0, h, tile_size):
                    tile = slide.read_region((x, y), l, (tile_size, tile_size))
                    tile = tile.convert('RGB')
                    tile = np.array(tile, dtype=np.float32)

                    tile /= tile.max()  # rescale to [0, 1]

                    # check if tile is valid (avoid white images & images with a low variance):
                    if np.sum(tile == 1 / np.sum(tile)) < white_threshold and np.std(tile) > std_threshold:
                        # remove black rows/columns:
                        non_zero_rows = np.any(tile != 0, axis=(1, 2))
                        non_zero_columns = np.any(tile != 0, axis=(0, 2))
                        tile = tile[non_zero_rows][:, non_zero_columns]
                        tiles.append(np.array(tile))

        return tiles
