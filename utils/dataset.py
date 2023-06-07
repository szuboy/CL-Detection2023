# !/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import numpy as np
import pandas as pd
from skimage import io as sk_io
from torch.utils.data import Dataset, DataLoader


class CephXrayDataset(Dataset):
    def __init__(self, csv_file_path, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.landmarks_frame = pd.read_csv(csv_file_path)
        self.transform = transform

    def __getitem__(self, index):
        image_file_path = str(self.landmarks_frame.iloc[index, 0])
        image = sk_io.imread(image_file_path)

        landmarks = self.landmarks_frame.iloc[index, 2:].values.astype('float')
        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.landmarks_frame)

