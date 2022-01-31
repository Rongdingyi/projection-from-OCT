import numpy as np
import torch
import os
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms.transforms import Pad

class BMPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, part):
        assert part in ['train', 'val', 'test']
        fh = open(data_dir+part+'.txt', 'r')
        lines = []
        self.data_dir = data_dir
        self.oct_list, self.octa_list, self.label_list_b2, self.label_list_b3 = [],[],[],[]
        for line in fh:
            line = line.rstrip()
            lines.append(line)
        for i in range(len(lines)):
            for j in range(1,401):
                oct_path = os.path.join(self.data_dir+'OCTA_6M_OCT/'+str(lines[i])+'/',str(j)+'.bmp')
                self.oct_list.append(oct_path)
                label_full_path_b2 = os.path.join(self.data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCT(ILM_OPL)/'+str(lines[i])+'.bmp')
                label_full_path_b3 = os.path.join(self.data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCT(OPL_BM)/'+str(lines[i])+'.bmp')
                self.label_list_b2.append([i, j-1, label_full_path_b2])
                self.label_list_b3.append([i, j-1, label_full_path_b3])
        self.length = len(self.oct_list)

        ###transform
        self.transform_picture = torchvision.transforms.Compose([torchvision.transforms.Pad(padding=(32, 0),padding_mode='reflect'),
        torchvision.transforms.ToTensor()])
        self.transform_resize_picture = torchvision.transforms.Compose([torchvision.transforms.Pad(padding=(32, 0),padding_mode='reflect'),
        transforms.Resize((320, 232)), torchvision.transforms.RandomRotation(30, resample=False,expand=False,center=None),torchvision.transforms.ToTensor()])
        self.transform_label = torchvision.transforms.Compose([torchvision.transforms.Pad(padding=(32, 0),padding_mode='reflect')])
        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #label to tensor
        label_info_b2 = self.label_list_b2[index]
        label_img_b2 = np.array(self.transform_label(Image.open(label_info_b2[2]).convert('RGB')), dtype=np.float32)
        label_img_b2 = label_img_b2[np.newaxis, label_info_b2[1]]/255
        label_img_b2 = self.to_tensor(label_img_b2)

        label_info_b3 = self.label_list_b3[index]
        label_img_b3 = np.array(self.transform_label(Image.open(label_info_b3[2]).convert('RGB')), dtype=np.float32)
        label_img_b3 = label_img_b3[np.newaxis, label_info_b3[1]]/255
        label_img_b3 = self.to_tensor(label_img_b3)

        key = torch.tensor([label_info_b2[0], label_info_b2[0], label_info_b2[0]], dtype=torch.float)
        key = key.view(3,1,1)
        label_img = torch.cat((key, label_img_b2, label_img_b3), 2)
        #picture to tensor
        PIL_img = Image.open(self.oct_list[index]).convert('RGB')
        img = self.transform_picture(PIL_img)
        resize_img = self.transform_resize_picture(PIL_img)           
        return img, resize_img, label_img


def load_data_bmp(batch_size, data_dir):  
    train_dataset = BMPDataset(data_dir, 'train')
    val_dataset = BMPDataset(data_dir, 'val')
    test_dataset = BMPDataset(data_dir, 'test')

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4)

    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)

    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                           shuffle=False, num_workers=4)                                          
    return train_iter, val_iter, test_iter

def main():
    train_iter, val_iter, test_iter = load_data_bmp(batch_size=1,data_dir='/media/rong/file/OCT_and_OCTA/')
    for batch_idx, (data, resize, target) in enumerate(train_iter):
        print(target.shape)

if __name__ == '__main__':
    main()
