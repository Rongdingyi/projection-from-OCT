import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

def pixel_from_image(input, predict, height, weight, device, label_name):
    batch_size = input.size(0)
    predict = F.interpolate(torch.cat((((predict[:,:,0,:]*640 - 320)/320).unsqueeze(3), (((predict[:,:,0,:] + 0.5*predict[:,:,1,:])*640 -320)/320).unsqueeze(3)),3), 
    size=[height, weight], mode='bilinear', align_corners=True)
    # predict = F.interpolate(predict, size=[self.height, self.weight], mode='bilinear', align_corners=True)
    # predict = predict - 0.5
    predict = predict.squeeze(1)
    grid = torch.zeros((batch_size, height, weight, 2), requires_grad=True).to(device)
    grid[:,:,:,1] = predict
    grid[:,:,:,0] = torch.linspace(-1, 1, steps=400)
    outp = torch.zeros((batch_size, 3, height, weight), requires_grad=True).to(device)
    outp = F.grid_sample(input , grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if label_name == 'B1' or 'B2' or 'B3' or 'B4':
        mean = torch.zeros((batch_size, 3, weight), requires_grad=True).to(device)
        mean = torch.mean(outp, dim=2, keepdim=False, out=None)
        result = mean.unsqueeze(dim=2).to(device)
    elif label_name == 'B5' or 'B6':
        max = torch.zeros((batch_size, 3, weight), requires_grad=True).to(device)
        max, index = torch.max(outp, dim=2, keepdim=False, out=None)
        result = max.unsqueeze(dim=2).to(device)        
    return predict, result