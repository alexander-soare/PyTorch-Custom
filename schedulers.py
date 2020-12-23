from torch.optim.lr_scheduler import _LRScheduler


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