import torch
from torch import LongTensor, FloatTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# TODO needs an overhaul
# 1. Consistency
# 2. See if we can make use of pytorch-metric-learning instead of these


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, with_logits=True, reduction: str = 'mean'):
        """
        https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py
        https://arxiv.org/pdf/1708.02002.pdf
        default gamma from tabel 1(b)
        """
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits

    def forward(self, input, target):
        """
        input to be (B, 1) of probabilites (not logits)
        """
        if self.with_logits:
            input = F.sigmoid(input)
        p_t = torch.where(target == 1, input, 1-input)
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t)
        # https://github.com/mathiaszinnen/focal_loss_torch/issues/2
        fl = torch.where(target == 1, fl * self.alpha, fl * (1 - self.alpha))
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x


class CategoricalFocalLoss(nn.Module):
    def __init__(self, gamma:float=2., reduction:str='mean'):
        """
        https://arxiv.org/pdf/1708.02002.pdf
        default gamma from table 1(b)
        we could specify alpha as a per class weighting but I haven't implemented
         it yet
        """
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input: FloatTensor, target: LongTensor):
        """
        expect input to be (B, N) where N is number of classes
        expect target to be (B, 1) If target is B, unsqueeze
        """
        target = target.reshape(target.shape[0], -1)
        p = torch.softmax(input, -1)
        p = torch.gather(p, -1, target)
        logp = torch.log_softmax(input, -1)
        logp = torch.gather(logp, -1, target)
        fl = - 1 * (1 - p) ** self.gamma * logp
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

    def forward(self, input, target):
        """ expects target to be categorical, and input to be logits
        """
        x = input.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * x, dim=self.dim))


class BCELabelSmoothingLoss(nn.Module):
    """
    Binary Cross Entropy Loss with label smoothing, takes logits
    """
    def __init__(self, smoothing):
        """
        `smoothing` is the smoothing factor. How much less confident than 100%
         are we on true labels?
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        """ expects target to be binary, and input to be logits
        """
        with torch.no_grad():
            target = torch.abs(target - self.smoothing)
        return F.binary_cross_entropy_with_logits(input, target)


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