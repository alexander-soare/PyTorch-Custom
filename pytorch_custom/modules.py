from torch import nn


class Identity(nn.Module):
    """
    just a helper for replacing a layer with identity
    as a way of "deleting" it
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x