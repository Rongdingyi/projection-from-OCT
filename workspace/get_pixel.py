from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats.morestats import circmean
import torch
import torchvision
import torch.nn.functional as F

number = 0
image = Image.open(os.path.join('/media/rong/file/OCT and OCTA/OCTA_6M_OCTA/10001/'+str(number+1)+'.bmp')).convert('RGB')
image = np.array(image)
# image = image.astype(np.float64)
label = Image.open(os.path.join('/media/rong/file/OCT and OCTA/OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/10001.bmp')).convert('RGB')
label = np.array(label)
target = label[399-number,:,:]
# print(target[0,:])
# print(image[0,0,:])
for i in range(400):
    n = 0
    for j in range(640):
        a = image[j,i,:]
        b = target[i,:]
        # print(a[0],b[0])
        if a[0] == b[0]:
            n += 1
            image[j,i,:] = [255,0,0]
plt.imshow(image[:,:,:])
plt.show()
    # print(b)
    # print(n)