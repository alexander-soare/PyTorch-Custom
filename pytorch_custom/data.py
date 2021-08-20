from itertools import islice

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import albumentations as A
import cv2

from .image_utils import normalize


class ImageDataset(torch.utils.data.Dataset):
    """
    most basic dataset for image task
    """
    def __init__(self, df, transforms=[]):
        self.file_paths = list(df.file_path)
        self.labels = list(df.label)
        if len(transforms):
            self.transforms = A.Compose(transforms)
        else:
            self.transforms = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, ix):
        target = self.labels[ix]
        img = self.load_image(self.file_paths[ix])
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = normalize(img)
        return {'img': torch.tensor(img).permute(2,0,1).float(),
                'target': torch.tensor(target)}

    def load_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class DataLoaders:
    """
    Convenience class for making data loaders for training and optionally also
     validation. Built in k-fold, stratified k-fold, and weighted sampling
     functionality
    Expects df with at least a file_path col
    If stratified_split=True, then expects a stratify group column
    If do_balance=True, expects a sampling_weight column where this weight refers
     to how that type of row should be sampled
    """
    def __init__(self, dataset_class, df, train_batch_size, train_dataset_args=[],
                    train_dataset_kwargs={}, val_dataset_args=[], 
                    val_dataset_kwargs = {}, val_batch_size=None, num_folds=0,
                    fold_ix=0, split_seed=None, do_stratify=True,
                    train_transforms=[], val_transforms=[], do_balance=False,
                    num_workers=0):
        
        if val_batch_size is None and num_folds > 0:
            val_batch_size = train_batch_size

        if num_folds > 0:
            kfold_class = StratifiedKFold if do_stratify else KFold
            kf = kfold_class(n_splits=num_folds, shuffle=True,
                                random_state=split_seed)
            if do_stratify:
                train_ix, val_ix = next(islice(kf.split(df.file_path,
                            df.stratify_group), fold_ix, fold_ix+1))
            else:
                train_ix, val_ix = next(islice(kf.split(
                                        df.file_path), fold_ix, fold_ix+1))
        else:
            train_ix = np.arange(len(df))
            
        self.train_dataset = dataset_class(df.iloc[train_ix],
                                            *train_dataset_args,
                                            transforms=train_transforms,
                                            **train_dataset_kwargs)
        if do_balance:
            train_sampler = WeightedRandomSampler(
                        torch.Tensor(list(df.iloc[train_ix].sampling_weight)),
                        len(self.train_dataset))
        else:
            train_sampler = None

        self.train_loader = DataLoader(self.train_dataset,
                        batch_size=train_batch_size, shuffle=(not do_balance),
                        num_workers=num_workers, pin_memory=True,
                        sampler=train_sampler,
                        collate_fn=getattr(self.train_dataset, 'collate_fn', None))


        if num_folds > 0:
            self.val_dataset = dataset_class(df.iloc[val_ix],
                                            *val_dataset_args,
                                            transforms=val_transforms,
                                            **val_dataset_kwargs)
            if do_balance:
                val_sampler = WeightedRandomSampler(
                        torch.Tensor(list(df.iloc[val_ix].sampling_weight)),
                        len(self.val_dataset))
            else:
                val_sampler = None
            self.val_loader = DataLoader(self.val_dataset,
                                    batch_size=val_batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True,
                                    sampler=val_sampler, 
                                    collate_fn=getattr(self.val_dataset, 'collate_fn', None))