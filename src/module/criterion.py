import torch
import torch.nn as nn
from docopt import Optional


class LikelihoodLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None,
                 reduction: str = 'mean'):
        super(LikelihoodLoss, self).__init__()
        self.basic_module = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)

    def __call__(self, prediction, targets):
        prediction = prediction.squeeze()
        assert prediction.shape == targets.shape
        return self.basic_module(prediction, targets)
