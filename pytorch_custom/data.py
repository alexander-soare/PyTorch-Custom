from itertools import islice
import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import albumentations as A
import cv2

from .image_utils import normalize


class ImageDataset(torch.utils.data.Dataset):
    """
    Most basic dataset for image task
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
        file_path = self.file_paths[ix]
        img = self.load_image(file_path)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        img = normalize(img)
        return {
            'img': torch.tensor(img).permute(2,0,1).float(),
            'target': torch.tensor(target),
            'file_path': file_path,
        }

    def load_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class DataLoaders:
    """
    Convenience class for making data loaders for training and optionally also
    validation. Built in k-fold, stratified k-fold, and weighted sampling
    functionality
    Expects df with at least a file_path col. We consider **unique** file  paths
    meaning splits are done according to file path not df row.
    If do_stratify=True, then expects kfold_stratum column.
    If do_group=True, then expects a kfold_group column
    If do_balance=True, expects a sampling_weight column where this weight refers
    to how that type of row should be sampled
    collate_fn will normally be grabbed from the dataset, but this behaviour
    may be overridden by providing one here.
    """
    def __init__(
            self, dataset_class, df, train_batch_size, train_dataset_args=[],
            train_dataset_kwargs={}, val_dataset_args=[],
            val_dataset_kwargs = {}, val_batch_size=None, num_folds=0,
            fold_ix=0, split_seed=None, do_stratify=True, do_group=False,
            train_transforms=[],
            val_transforms=[], do_balance_train=False, do_balance_val=False,
            num_workers=0, train_collate_fn=None, val_collate_fn=None,
            do_shuffle_train_data=True):
        
        if val_batch_size is None and num_folds > 0:
            val_batch_size = train_batch_size

        file_paths = np.array(df.file_path.unique())
        if num_folds > 0:
            if do_stratify:
                # NOTE: All this kfold_stratum_header juggling is for BC
                kfold_stratum_header = 'kfold_stratum'
                if 'kfold_stratum' not in df.columns:
                    if 'stratify_group' in df.columns:
                        kfold_stratum_header = 'stratified_group'
                        warnings.warn(
                            ("Deprecation Warning: The use of 'stratify_group' "
                             "as the df colum header for stratified kfold will "
                             "be deprecated. Please use `kfold_stratum` instead."))

                assert len(df.drop_duplicates(
                    subset=['file_path', kfold_stratum_header])) \
                    == len(file_paths), ("It is required that file_path and "
                                         f"{kfold_stratum_header} are uniquely paired "
                                         "for a stratified split.")
            if do_group:
                assert len(df.drop_duplicates(
                    subset=['file_path', 'kfold_group'])) \
                    == len(file_paths), ("It is required that file_path and "
                                         "kfold_group are uniquely paired "
                                         "for a grouped split.")

            if do_stratify and do_group:
                kfold_class = StratifiedGroupKFold
                dedupe = df.drop_duplicates(
                    subset=['file_path', kfold_stratum_header, 'kfold_group'])
                split_kwargs = dict(
                    X = dedupe.file_path,
                    y = dedupe[kfold_stratum_header],
                    groups = dedupe.kfold_group
                )
            elif do_stratify:
                kfold_class = StratifiedKFold
                dedupe = df.drop_duplicates(
                    subset=['file_path', kfold_stratum_header])
                split_kwargs = dict(
                    X = dedupe.file_path,
                    y = dedupe[kfold_stratum_header],
                )
            elif do_group:
                kfold_class = GroupKFold
                dedupe = df.drop_duplicates(
                    subset=['file_path', 'kfold_group'])
                split_kwargs = dict(
                    X = dedupe.file_path,
                    groups = dedupe['kfold_group'],
                )
            else:
                kfold_class = KFold
                split_kwargs = dict(
                    X = df.file_path,
                )
            extra_kwargs = {}
            if kfold_class != GroupKFold:
                extra_kwargs['shuffle'] = True
                extra_kwargs['random_state'] = split_seed
            else:
                warnings.warn("Note that for GroupKFold there is not shuffle option")
            kf = kfold_class(n_splits=num_folds, **extra_kwargs)
            
            train_ix, val_ix = next(islice(kf.split(
                **split_kwargs), fold_ix, fold_ix+1))
            train_file_paths = set(file_paths[train_ix])
            val_file_paths = set(file_paths[val_ix])
        else:
            train_file_paths = file_paths
            
        self.train_dataset = dataset_class(
            df[df.file_path.isin(train_file_paths)], *train_dataset_args,
            transforms=train_transforms, **train_dataset_kwargs)
        if do_balance_train:
            train_sampler = WeightedRandomSampler(
                torch.Tensor(list(df[
                    df.file_path.isin(train_file_paths)].sampling_weight)),
                len(self.train_dataset))
        else:
            train_sampler = None

        train_collate_fn = train_collate_fn or getattr(
            self.train_dataset, 'collate_fn', None)
        self.train_loader = DataLoader(self.train_dataset,
            batch_size=train_batch_size,
            shuffle=(not do_balance_train and do_shuffle_train_data),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler,
            collate_fn=train_collate_fn)

        if num_folds > 0:
            self.val_dataset = dataset_class(
                df[df.file_path.isin(val_file_paths)], *val_dataset_args,
                transforms=val_transforms, **val_dataset_kwargs)
            if do_balance_val:
                val_sampler = WeightedRandomSampler(
                    torch.Tensor(list(df[
                        df.file_path.isin(val_file_paths)].sampling_weight)),
                    len(self.val_dataset))
            else:
                val_sampler = None
            val_collate_fn = val_collate_fn or getattr(
                self.val_dataset, 'collate_fn', None)
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=val_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, sampler=val_sampler, 
                collate_fn=val_collate_fn)