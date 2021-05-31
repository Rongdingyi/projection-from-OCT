from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

data_dir = '/media/rong/file/OCT and OCTA/'
fh = open(data_dir+'train.txt', 'r')
lines = []

for line in fh:
    line = line.rstrip()
    lines.append(line)
images = np.zeros((len(lines)*400,640,400,3),int)
label_image = np.zeros((len(lines)*400,400,3),int)
for i in range(len(lines)):
    label = Image.open(os.path.join(data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/'+str(lines[i])+'.bmp')).convert('RGB')
    label = np.array(label)
    for j in range(400):
        image_path = str(j + 1) + ".bmp"
        picture = Image.open(os.path.join(data_dir+'OCTA_6M_OCTA/'+lines[i]+'/'+image_path)).convert('RGB')
        pic = np.array(picture)
        images[400*i+j,:,:,:] = pic
        label_image[400*i+j,:,:] = label[399-j,:,:]
image = images[0,:,:,:]
label = label_image[0,:,:]
label = label[np.newaxis,:,:]
print(image.shape)
print(label.shape)

# print(images.shape)