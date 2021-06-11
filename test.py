import os
from models.models import ResNet18, ResNet50
from dataset.octa_dataset import load_data_bmp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.loss import MaxPoolLoss
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from models.projection import pixel_from_image

def main(input_root, output_root, picture_name, label_name, end_epoch):
    n_channels = 3
    dir_path = os.path.join(output_root, 'checkpoints_b21')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_loader, val_loader, test_loader = load_data_bmp(batch_size=4, data_dir=input_root, picture='OCT', label='B2')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels).to(device)
    model = torch.nn.DataParallel(model)
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_loss_%.5f.pth' % (18, 0.00950))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, test_loader, device, label_name, output_root=output_root)

def test(model, data_loader, device, label_name, output_root=None):
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            batch = inputs.size(0)
            targets = targets.to(device)
            predict = model(inputs.to(device)).to(device)
            # targets, outputs = targets.cpu().numpy(), outputs.cpu().numpy()
            pixel_coord, result = pixel_from_image(inputs.to(device), predict, 2, 400, device, label_name)
            pixel_coord = pixel_coord * 320 + 320
            inputs, pixel_coord = inputs.cpu().numpy(), pixel_coord.cpu().numpy()
            inputs = (inputs * 255)
            # print(pixel_coord[0,0,:])
            # print(pixel_coord.shape)
            y = np.linspace(0, 400, num=400)
            # loss = criterion(inputs.to(device), predict, targets, device)
            for i in range(batch):
                plt.cla()
                plt.imshow(inputs[i,0,:,:])
                plt.plot(y,pixel_coord[i,0,:],'y-')
                plt.plot(y,pixel_coord[i,1,:],'r-')
                imgfile = '/media/rong/file/predict_img/'+str(batch_idx*batch+i)+'.jpg'
                plt.savefig(imgfile)




if __name__ == '__main__':
    main(input_root='/media/rong/file/OCT and OCTA/', output_root='./output', picture_name='OCT', label_name='B2', end_epoch=100)