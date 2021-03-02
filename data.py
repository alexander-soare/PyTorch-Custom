from itertools import islice

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader


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
    def __init__(self, dataset_class, df, train_batch_size,
                    train_dataset_kwargs={}, val_dataset_kwargs={}, 
                    val_batch_size=None, num_folds=0, fold_ix=0, split_seed=None,
                    do_stratify=True, train_transforms=[], val_transforms=[],
                    do_balance=False, num_workers=0):
        
        if val_batch_size is None and num_folds > 0:
            val_batch_size = train_batch_size

        kfold_class = StratifiedKFold if do_stratify else KFold

        if num_folds > 0:
            kf = kfold_class(n_splits=num_folds, shuffle=True,
                                random_state=split_seed)
            train_ix, val_ix = next(islice(kf.split(df.file_path,
                            df.stratify_group), fold_ix, fold_ix+1))
        else:
            train_ix = np.arange(len(df))
            
        self.train_dataset = dataset_class(df.iloc[train_ix],
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
                        sampler=train_sampler)


        if num_folds > 0:
            self.val_dataset = dataset_class(df.iloc[val_ix],
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
                                    sampler=val_sampler)