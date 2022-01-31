import os
from dataset.dataset import load_data_bmp
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from models.projection import projections
from models.model import UNet
import numpy as np
import math
import argparse
from torchvision import transforms

def PSNR(img1, img2):
    mse = np.mean((img1 - img2)** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def main(input_root, output_root, checkpoint_path, idx):
    for i in range(10201, 10301):
        with open(input_root+'test.txt','a+',encoding='utf-8') as fi:
            fi.truncate(0)
            fi.write(str(i)+'\n')
        train_loader, val_loader, test_loader = load_data_bmp(batch_size=1, data_dir=input_root)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet(n_channels=3, n_classes = 4).to(device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint_path)['net'])
        eval(model, test_loader, device, number=i, output_root=output_root)
        eval_seg(model, test_loader, device, number=i, idx=idx, output_root=output_root)

def eval(model, data_loader, device, number, output_root):
    model.eval()
    B2, B2_gt = np.zeros((400,464,3)), np.zeros((400,464,3))
    B3, B3_gt = np.zeros((400,464,3)), np.zeros((400,464,3))
    with torch.no_grad():
        for batch_idx, (or_inputs, inputs, targets) in enumerate(data_loader):
            or_inputs = or_inputs.to(device)
            predict = model(inputs.to(device)).to(device)
            coor_b2, coor_b3, result_b2, result_b3 = projections(or_inputs, predict, 64, 464, device)
            result_b2, result_b3 = result_b2.squeeze(0).cpu().numpy().transpose(1,2,0), result_b3.squeeze(0).cpu().numpy().transpose(1,2,0)
            gt_b2, gt_b3 = targets[:,:,:,1:465].squeeze(0).cpu().numpy().transpose(1,2,0), targets[:,:,:,465:929].squeeze(0).cpu().numpy().transpose(1,2,0)
            B2[399-batch_idx,:,:], B2_gt[399-batch_idx,:,:] = result_b2[0,:,:], gt_b2[0,:,:]
            B3[399-batch_idx,:,:], B3_gt[399-batch_idx,:,:] = result_b3[0,:,:], gt_b3[0,:,:]
    min_b2, max_b2 = np.min(B2[:,32:432,0]), np.max(B2[:,32:432,0])
    B2 = (B2 - min_b2)/(max_b2-min_b2)
    min_b3, max_b3 = np.min(B3[:,32:432,0]), np.max(B3[:,32:432,0])
    B3 = (B3 - min_b3)/(max_b3-min_b3)            

    imgfile_root = output_root + '/predict_picture'
    if not os.path.exists(imgfile_root):
        os.makedirs(imgfile_root)
    imgfile_b2 = imgfile_root + '/predict_'+ str(number)+'_'+'b2.bmp' 
    imgfile_b3 = imgfile_root + '/predict_'+ str(number)+'_'+'b3.bmp' 
    matplotlib.image.imsave(imgfile_b2, B2[:,32:432,:])
    matplotlib.image.imsave(imgfile_b3, B3[:,32:432,:])
    B2_psnr, B3_psnr = PSNR(B2_gt[:,32:432,:], B2[:,32:432,:]), PSNR(B3_gt[:,32:432,:], B3[:,32:432,:])
    f = open("{}/psnr_{}.txt".format(imgfile_root, number), "w")
    f.write(str([B2_psnr, B3_psnr]))
    f.close()
    print(B2_psnr, B3_psnr)

def eval_seg(model, data_loader, device, number, idx, output_root):
    model.eval()
    with torch.no_grad():
        for batch_idx, (or_inputs, inputs, targets) in enumerate(data_loader):
            if batch_idx == idx:
                targets = targets.to(device)
                predict = model(inputs.to(device)).to(device)
                coor_b2, coor_b3, result_b2, result_b3 = projections(or_inputs.to(device), predict, 5, 464, device)
                coor_b2, coor_b3 = coor_b2 * 320 + 320, coor_b3 * 320 + 320
                coor_b2, coor_b3 = coor_b2.cpu().numpy(), coor_b3.cpu().numpy()
                or_inputs = np.array(transforms.ToPILImage()(or_inputs[0,:,:,:]))
                y = np.linspace(0, 400, num=400)
                plt.cla()
                plt.imshow(or_inputs[:,32:432,:])
                plt.plot(y,coor_b2[0,0,32:432],'r-', y,coor_b2[0,-1,32:432],'y-', y,coor_b3[0,0,32:432],'g-',  coor_b3[0,-1,32:432],'b-')
                plt.axis('off')
                imgfile_root = output_root+'/predict_picture_single3'
                if not os.path.exists(imgfile_root):
                    os.makedirs(imgfile_root)
                imgfile = imgfile_root + '/'+ str(number)+'_'+str(idx)+'.png' 
                plt.savefig(imgfile, bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN model')
    parser.add_argument('--input_root',
                        default='/data1/rong/OCT_and_OCTA/',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--checkpoint_path',
                        default='./checkpoints',
                        help='checkpoint root',
                        type=str)     
    parser.add_argument('--idx',
                        default='3',
                        help='which slice to use',
                        type=str)                    

    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    checkpoint_path = args.checkpoint_path
    idx = args.idx
    main(input_root, output_root, checkpoint_path, idx)