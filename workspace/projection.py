from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats.morestats import circmean
import torch
import torchvision
import torch.nn.functional as F
from models.loss import MaxPoolLoss

number = 59
image = Image.open(os.path.join('/media/rong/file/OCT and OCTA/OCTA_6M_OCTA/10001/'+str(number+1)+'.bmp')).convert('RGB')
image = np.array(image)
# image = image.astype(np.float64)
label = Image.open(os.path.join('/media/rong/file/OCT and OCTA/OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/10001.bmp')).convert('RGB')
label = np.array(label)
target = np.zeros([1,400,3])
target[0,:,:] = label[399 - number,:,:]
predict = np.array([[219.6,229.0,240.9,261.3,267.3,263.0,273.2,284.3,286.0,284.3,286.8,280.9,280.0,280.6,280.9,284.3,286.0,284.3,273.2,263.0,267.3,261.3,240.9,229.0,219.6],
           [258.8,269.0,280.6,300.3,307.3,303.0,313.2,324.3,326.0,324.3,326.8,320.9,320.0,320.6,320.9,324.3,326.0,324.3,313.2,303.0,307.3,301.0,280.9,269.0,259.6]])
tansform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
image = tansform(image).unsqueeze(dim=0)
print(image[:,:,200:250,:])
channel = image.size(0)
target = tansform(target).unsqueeze(dim=0)
predict = tansform(predict).unsqueeze(dim=0)
# criterion = MaxPoolLoss(10 ,400)
# loss = criterion(image, predict, target)
# maxtrice = torch.zeros(2,10,400,2)
# maxtrice[:,:,:,0] = torch.linspace(-1, 1, steps=400)
predict = torch.nn.functional.interpolate(predict, size=[10,400], mode='bicubic', align_corners=True)
predict = (predict - 320)/320
grid = torch.zeros(1, 10, 400, 2)
grid[:,:,:,1], grid[:,:,:,0] = predict, torch.linspace(-1, 1, steps=400)
outp = F.grid_sample(image , grid, mode='bilinear', padding_mode='zeros', align_corners=True)


max, index = torch.max(outp, dim=2, keepdim=False, out=None)
# max = max.unsqueeze(dim=2)
# loss = F.smooth_l1_loss(max, target)
# print(loss)


# print(outp*255)
# outp = outp.squeeze(dim=0)
# outp = outp*255
# outp = outp.cpu().detach().numpy() 
# print(outp)
# image1 = Image.open(os.path.join('/media/rong/file/OCT and OCTA/OCTA_6M_OCTA/10001/'+str(number+1)+'.bmp')).convert('RGB')
# image1 = np.array(image1)
# print(image1[350,399,:])