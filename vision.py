from torch import nn
import torch.nn.functional as F

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


class Identity(nn.Module):
    """
    Just a way of easily "deleting" layers in pretrained models
    """
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp


class ResnetBase(nn.Module):
    def __init__(self, backbone_func, n_classes, backbone_args=[],
                    backbone_kwargs={}, dropout=0, feature_pooler=None):
        """
        Expects Resnet backbone function (soon to be expanded)
        Allows replacing the following:
         - final feature map pooling operation (defaults to average pool)
         - dropout after pooling (defaults to 0)
         - `num_classes` number of output classes (must be provided)
        """
        super().__init__()
        self.backbone = backbone_func(*backbone_args, **backbone_kwargs)
        if feature_pooler is not None:
            self.backbone.avgpool = feature_pooler
        self.dropout = nn.Dropout(p=dropout)

        # make a new fc layer
        self.fc = nn.Linear(self.backbone.fc.in_features, n_classes)
        # drop the backbone's fc layer
        self.backbone.fc = Identity()
        self.eval()        

    def forward(self, inp):
        # forward through backbone
        x = self.backbone(inp)
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