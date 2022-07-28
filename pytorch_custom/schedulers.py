from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class WarmupExponentialLR(_LRScheduler):
    def __init__(self, optimizer, start_lr, warmup_itrs, gamma, warmup_type='exp'):
        assert warmup_type in [
            'lin', 'exp'], f"warmup_type == {warmup_type} not implemented"
        self.start_lr = start_lr
        self.warmup_itrs = warmup_itrs
        self.warmup_type = warmup_type
        self.gamma = gamma
        self.itr = 0
        self.default_lrs = []
        self.optimizer = optimizer
        for param_group in self.optimizer.param_groups:
            self.default_lrs.append(param_group['lr'])
        super(WarmupExponentialLR, self).__init__(optimizer)

    def step(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if self.itr < self.warmup_itrs:
                if self.warmup_type == 'exp':
                    new_lr = self.start_lr * \
                        (self.default_lrs[i] /
                         self.start_lr)**(self.itr/self.warmup_itrs)
                if self.warmup_type == 'lin':
                    new_lr = self.start_lr + \
                        (self.default_lrs[i] - self.start_lr) * \
                        (self.itr/self.warmup_itrs)
                param_group['lr'] = new_lr
            elif self.itr == self.warmup_itrs:
                param_group['lr'] = self.default_lrs[i]
            else:
                param_group['lr'] = self.default_lrs[i] * \
                    self.gamma ** (self.itr - self.warmup_itrs)
        self.itr += 1


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    """
    Subclassing torch.optim.lr_scheduler.ReduceLROnPlateau
    added warmup parameters
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, warmup_itrs=0, warmup_type='lin',
                 start_lr=1e-16, start_patience=0, verbose=False):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience,
                 threshold=threshold, threshold_mode=threshold_mode,
                 cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)
        self.warmup_itrs = warmup_itrs
        self.warmup_type = warmup_type
        self.start_lr = start_lr
        self.default_lrs = []
        self.itr = 0
        for param_group in optimizer.param_groups:
            self.default_lrs.append(param_group['lr'])
        self.start_patience = start_patience

    def step(self, metrics):
        if self.itr < self.warmup_itrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if self.warmup_type == 'exp':
                    new_lr = self.start_lr * \
                        (self.default_lrs[i] /
                         self.start_lr)**(self.itr/self.warmup_itrs)
                if self.warmup_type == 'lin':
                    new_lr = self.start_lr + \
                        (self.default_lrs[i] - self.start_lr) * \
                        (self.itr/self.warmup_itrs)
                param_group['lr'] = new_lr
        elif self.itr == self.warmup_itrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.default_lrs[i]
        elif self.itr < self.start_patience:
            pass
        else:
            super().step(metrics)
        self.itr += 1