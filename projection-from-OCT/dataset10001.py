import numpy as np
import torch
import os
import torch
from PIL import Image
import torchvision

class BMPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, part):
        assert part in ["train", "val"]
        fh = open(data_dir+part+'.txt', 'r')
        lines = []
        for line in fh:
            line = line.rstrip()
            lines.append(line)
        self.length = len(lines)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.images , self.labels = get_image_label(data_dir, part)

    def __len__(self):
        return 400*self.length

    def __getitem__(self, index):
        img = self.images[index,:,:,:]
        label = self.labels[index,:,:]
        label = label[np.newaxis,:,:]
        img = self.transform(img)
        label = self.transform(label)
        return img, label

def get_image_label(data_dir, part):
    assert part in ["train", "val"]
    fh = open(data_dir+part+'.txt', 'r')
    lines = []
    for line in fh:
        line = line.rstrip()
        lines.append(line)
    images = np.zeros((len(lines)*400,640,400,3),dtype='float32')
    label_image = np.zeros((len(lines)*400,400,3),dtype='float32')
    for i in range(len(lines)):
        label = Image.open(os.path.join(data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/'+str(lines[i])+'.bmp')).convert('RGB')
        label = np.array(label)
        for j in range(400):
            image_path = str(j + 1) + ".bmp"
            picture = Image.open(os.path.join(data_dir+'OCTA_6M_OCTA/'+lines[i]+'/'+image_path)).convert('RGB')
            pic = np.array(picture)
            images[400*i+j,:,:,:] = pic/255
            label_image[400*i+j,:,:] = label[399-j,:,:]
    return images, label_image

def load_data_bmp(batch_size, data_dir):  
    train_dataset = BMPDataset(data_dir, 'train')
    val_dataset = BMPDataset(data_dir, 'val')


    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4)

    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)
    return train_iter, val_iter

def main():
    train_iter, val_iter = load_data_bmp(batch_size=1,data_dir='/media/rong/file/OCT and OCTA/')
    for batch_idx, (data, target) in enumerate(train_iter):
        y = data
        print(data[0,0,200:250,:])

if __name__ == '__main__':
    main()
