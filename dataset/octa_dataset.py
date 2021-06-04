import numpy as np
import torch
import os
import torch
from PIL import Image
import torchvision

# class BMPDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, part):
#         assert part in ["train", "val"]
#         fh = open(data_dir+part+'.txt', 'r')
#         lines = []
#         self.data_dir = data_dir
#         for line in fh:
#             line = line.rstrip()
#             lines.append(line)
#         self.length = len(lines)
#         self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#         self.images , self.labels = get_image_label(data_dir, part)

#     def __len__(self):
#         return 400*self.length

#     def __getitem__(self, index):
#         image_path = str(index + 1) + ".bmp"
#         PIL_img = Image.open(os.path.join(self.data_dir+'OCTA_6M_OCTA/10001'+'/'+image_path)).convert('RGB')
#         img = self.transform(PIL_img)
#         label = self.labels[index,:,:]
#         label = label[np.newaxis,:,:]
#         label = self.transform(label)
#         return img, label

# def get_image_label(data_dir, part):
#     assert part in ["train", "val"]
#     fh = open(data_dir+part+'.txt', 'r')
#     lines = []
#     for line in fh:
#         line = line.rstrip()
#         lines.append(line)
#     images = np.zeros((len(lines)*400,640,400,3),dtype='float32')
#     label_image = np.zeros((len(lines)*400,400,3),dtype='float32')
#     for i in range(len(lines)):
#         label = Image.open(os.path.join(data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/'+str(lines[i])+'.bmp')).convert('RGB')
#         label = np.array(label)
#         for j in range(400):
#             image_path = str(j + 1) + ".bmp"
#             picture = Image.open(os.path.join(data_dir+'OCTA_6M_OCTA/'+lines[i]+'/'+image_path)).convert('RGB')
#             pic = np.array(picture)
#             images[400*i+j,:,:,:] = pic/255
#             label_image[400*i+j,:,:] = label[399-j,:,:]/255
#     return images, label_image
class BMPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, part):
        assert part in ["train", "val"]
        fh = open(data_dir+part+'.txt', 'r')
        lines = []
        self.data_dir = data_dir
        self.img_list, self.mask_list = [],[]
        for line in fh:
            line = line.rstrip()
            lines.append(line)
            for i in range(1,401):
                img_path = os.path.join(self.data_dir+'OCTA_6M_OCTA/'+str(line)+'/',str(i)+'.bmp')
                self.img_list.append(img_path)
        self.length = len(lines)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.labels = get_image_label(data_dir, part)

    def __len__(self):
        return 400*self.length

    def __getitem__(self, index):
        PIL_img = Image.open(self.img_list[index]).convert('RGB')
        img = self.transform(PIL_img)
        label = self.labels[index,:,:]
        label = label[np.newaxis,:,:]
        label = self.transform(label)
        return img, label

def get_image_label(data_dir, part):
    assert part in ["train", "val"]
    fh = open(data_dir+part+'.txt', 'r')
    lines = []
    for line in fh:
        line = line.rstrip()
        lines.append(line)
    label_image = np.zeros((len(lines)*400,400,3),dtype='float32')
    for i in range(len(lines)):
        label = Image.open(os.path.join(data_dir+'OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/'+str(lines[i])+'.bmp')).convert('RGB')
        label = np.array(label)
        for j in range(400):
            label_image[400*i+j,:,:] = label[399-j,:,:]/255
    return label_image


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
        target = target.cpu().numpy()
        data = data.cpu().numpy()
        print(data.shape, target.shape)

if __name__ == '__main__':
    main()
