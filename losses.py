import torch
import torch.nn as nn

# https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py
class FocalLoss(nn.Module):
    """
    I've modified this in the following ways:
    1. with_logits logic. WARNING - not yet tested for multi-class
    2. with_binary logic
    """
    def __init__(self, alpha=1, gamma=2, with_logits=False, binary=False,
                    reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
        self.binary = binary

    def forward(self, y_pred, y_true):
        if self.with_logits:
            if self.binary:
                y_pred = torch.sigmoid(y_pred)
            else:
                # WARNING not tested yet
                y_pred = torch.softmax(y_pred, -1)
        # WARNING this assumes we are not doing label smoothing
        p_t = torch.where(y_true == 1, y_pred, 1-y_pred)
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        fl = torch.where(y_true == 1, fl * self.alpha, fl)
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x