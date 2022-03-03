import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform = None, mode: bool = True, config = None):
        self.df = df
        self.transform = transform
        self.mode = mode

        if self.mode:
            self.image_paths = self.df['path'].tolist()
            self.targets = self.df[config.target_col].tolist()
            self.split_targets = self.df[config.split_target_col].tolist()
        else:
            self.image_paths = [os.path.join(config.submission_image_dir, img_id) for img_id in self.df.ImageID]
            self.targets = [0] * len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.targets[index])

    def __len__(self):
        return len(self.image_paths)


class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)