import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from models.projection import pixel_from_image

class MaxPoolLoss(nn.Module):
    def __init__(self, height=10, weight=400):
        super(MaxPoolLoss, self).__init__()
        self.height = height
        self.weight = weight

    def forward(self, result, target):
        loss = F.smooth_l1_loss(result, target)
        return loss
