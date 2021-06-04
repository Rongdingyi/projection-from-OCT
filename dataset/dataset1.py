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
        return 102

    def __getitem__(self, index):
        img = self.images[index,:,:,:]
        label = self.labels[index,:,:]
        # label = label[np.newaxis,:,:]
        img = self.transform(img)
        # label = self.transform(label)
        return img, label

def get_image_label(data_dir, part):
    assert part in ["train", "val"]
    fh = open(data_dir+part+'.txt', 'r')
    lines = []
    for line in fh:
        line = line.rstrip()
        lines.append(line)
    images = np.zeros((102,640,400,3),dtype='float32')
    label_image = np.zeros((1,10,400),dtype='float32')
    for i in range(len(lines)):
        # label = Image.open(os.path.join(data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/10001.bmp')).convert('RGB')
        # label = np.array(label)
        for j in range(102):
            image_path = str(j + 1) + ".bmp"
            picture = Image.open(os.path.join(data_dir+'OCTA_6M_OCTA/'+lines[i]+'/'+image_path)).convert('RGB')
            pic = np.array(picture)
            images[400*i+j,:,:,:] = pic/255
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            predict = np.array([[219.6,229.0,240.9,261.3,267.3,263.0,273.2,284.3,286.0,284.3,286.8,280.9,280.0,280.6,280.9,284.3,286.0,284.3,273.2,263.0,267.3,261.3,240.9,229.0,219.6],
           [258.8,269.0,280.6,300.3,307.3,303.0,313.2,324.3,326.0,324.3,326.8,320.9,320.0,320.6,320.9,324.3,326.0,324.3,313.2,303.0,307.3,301.0,280.9,269.0,259.6]])
            predict = transform(predict)
            predict = predict.unsqueeze(0)
            predict = torch.nn.functional.interpolate(predict, size=[10,400], mode='bilinear', align_corners=True)
            predict = predict/640
            predict1 = predict.repeat(102, 1, 1, 1)
    return images, predict1

def load_data_bmp(batch_size, data_dir):  
    train_dataset = BMPDataset(data_dir, 'train')
    val_dataset = BMPDataset(data_dir, 'val')


    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4)

    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)
    return train_iter, val_iter

def main():
    train_iter, val_iter = load_data_bmp(batch_size=4,data_dir='/media/rong/file/OCT and OCTA/')
    for batch_idx, (data, target) in enumerate(train_iter):
        print(data.shape, target.shape)
        # print(target)

if __name__ == '__main__':
    main()
