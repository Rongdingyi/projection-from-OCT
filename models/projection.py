import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

def projections(input, coor_predict, height, weight, device):
    batch_size = input.size(0)
    baseline = torch.zeros((coor_predict.size(0), coor_predict.size(1), 2, coor_predict.size(3)), requires_grad=True).to(device)

    predict1 = torch.zeros((coor_predict.size(0), coor_predict.size(1), 2, coor_predict.size(3)), requires_grad=True).to(device)
    predict1[:,:,0,:] = baseline[:,:,0,:].detach() + coor_predict[:,:,0,:]
    predict1[:,:,1,:] = baseline[:,:,1,:].detach() + coor_predict[:,:,1,:]
    predict1 = F.interpolate(predict1, size=[height, weight], mode='bilinear', align_corners=True)
    predict1 = predict1.squeeze(1)
    grid1 = torch.zeros((batch_size, height, weight, 2), requires_grad=True).to(device)
    grid1[:,:,:,1] = predict1
    grid1[:,:,:,0] = torch.linspace(-1, 1, steps=weight) ##change
    outp1 = torch.zeros((batch_size, 3, height, weight), requires_grad=True).to(device)
    outp1 = F.grid_sample(input , grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
    mean1 = torch.zeros((batch_size, 3, weight), requires_grad=True).to(device)
    mean1 = torch.mean(outp1, dim=2, keepdim=False, out=None)
    result1 = mean1.unsqueeze(dim=2).to(device)

    predict2 = torch.zeros((coor_predict.size(0), coor_predict.size(1), 2, coor_predict.size(3)), requires_grad=True).to(device)
    predict2[:,:,0,:] = baseline[:,:,0,:].detach() + coor_predict[:,:,2,:]
    predict2[:,:,1,:] = baseline[:,:,1,:].detach() + coor_predict[:,:,3,:]
    predict2 = F.interpolate(predict2, size=[height, weight], mode='bilinear', align_corners=True)
    predict2 = predict2.squeeze(1)
    grid2 = torch.zeros((batch_size, height, weight, 2), requires_grad=True).to(device)
    grid2[:,:,:,1] = predict2
    grid2[:,:,:,0] = torch.linspace(-1, 1, steps=weight) ##change
    outp2 = torch.zeros((batch_size, 3, height, weight), requires_grad=True).to(device)
    outp2 = F.grid_sample(input , grid2, mode='bilinear', padding_mode='zeros', align_corners=True)
    mean2 = torch.zeros((batch_size, 3, weight), requires_grad=True).to(device)
    mean2 = torch.mean(outp2, dim=2, keepdim=False, out=None)
    result2 = mean2.unsqueeze(dim=2).to(device)     
    return outp1, result1, result1, result2