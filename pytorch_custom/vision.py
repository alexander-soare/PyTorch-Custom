import torch
from torch import nn
import torch.nn.functional as F

from .modules import Identity

# https://amaarora.github.io/2020/08/30/gempool.html
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p),
                (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' \
            + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' \
            + str(self.eps) + ')'


class ResnetBase(nn.Module):
    """
    Helpful bridge to torchvision's resnet.
    WARNING: We are susceptible to changes in the torchvision code. For now,
     this is the most relevant snippet
     https://github.com/pytorch/vision/blob/f637c63b0ce05328ffc6c98d0e1cb8e3ab1adaa4/torchvision/models/resnet.py#L230-L246
    """

    def __init__(self, backbone_func, n_classes, backbone_args=[],
                    backbone_kwargs={}, dropout=0,
                    feature_pooler=nn.AdaptiveAvgPool2d((1, 1)),
                    as_feature_extractor=False):
        """
        Expects Resnet backbone function (soon to be expanded)
        Allows replacing the following:
         - final feature map pooling operation (defaults to average pool)
         - dropout after pooling (defaults to 0)
         - `num_classes` number of output classes (must be provided)
         - `as_feature_extractor` lets us get the output of the backbone without
            fowarding through the fully connected layer. Hint: if you want the
            feature maps without any spatial aggregation, set feature_pooler as
            the identity mapping
        """
        super().__init__()
        self.backbone = backbone_func(*backbone_args, **backbone_kwargs)
        self.backbone.layer4.register_forward_hook(self._hook_feature_map)
        self.feature_pooler = feature_pooler
        self.as_feature_extractor = as_feature_extractor
        self.dropout = nn.Dropout(p=dropout)

        # good to keep track of this number
        self.n_features = self.backbone.fc.in_features
        # make a new fc layer
        self.fc = nn.Linear(self.n_features, n_classes)
        # drop the backbone's fc layer
        self.backbone.fc = Identity()
        self.eval()        

    def _hook_feature_map(self, *args):
        """ grabs the output of layer4
        """
        self.feature_map = args[-1]

    def forward(self, inp):
        # forward through backbone
        self.backbone(inp)
        x = self.feature_map
        x = self.feature_pooler(x)
        if self.as_feature_extractor:
            return x
        # flatten for fc
        x = torch.flatten(x, 1)
        # dropout
        x = self.dropout(x)
        # get logits
        x = self.fc(x)
        return x

    def load(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)