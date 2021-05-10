import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# TODO needs an overhaul
# 1. Consistency
# 2. See if we can make use of pytorch-metric-learning instead of these

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


# https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
class CELabelSmoothingLoss(nn.Module):
    """
    Cross Entropy Loss with label smoothing, takes logits
    """
    def __init__(self, smoothing, n_classes, dim=-1):
        """
        `n_classes` is number of classes
        `smoothing` is the smoothing factor. How much less confident than 100%
         are we on true labels?
        """
        super(CELabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        """ expects target to be categorical, and pred to be logits
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



class BCELabelSmoothingLoss(nn.Module):
    """
    Binary Cross Entropy Loss with label smoothing, takes logits
    """
    def __init__(self, smoothing):
        """
        `smoothing` is the smoothing factor. How much less confident than 100%
         are we on true labels?
        """
        super(BCELabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """ expects target to be binary, and pred to be logits
        """
        with torch.no_grad():
            target = torch.abs(target - self.smoothing)
        return F.binary_cross_entropy_with_logits(pred, target)


# https://discuss.pytorch.org/t/label-smoothing-with-ctcloss/103392
class SmoothCTCLoss(_Loss):
    """
    TODO I still need to do some work to properly understand this
    """
    def __init__(self, num_classes, blank=0, weight=0.01):
        super().__init__(reduction='mean')
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)

        #print(ctc_loss, kldiv_loss)
        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss