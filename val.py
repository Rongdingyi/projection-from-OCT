import os
from models.model import UNet
from dataset.dataset import load_data_bmp
import torch
import torch.optim as optim
from models.loss import MaxPoolLoss, PerceptualLoss
from tqdm import trange
from tqdm import tqdm
from utils import metrics
import argparse
from models.projection import pixel_from_image
from tensorboardX import SummaryWriter
from eval import test, test_copy
import torchvision
import numpy as np

def main(input_root, output_root, picture_name, label_name, end_epoch):
    writer = SummaryWriter()
    n_channels = 3
    lr = 0.0001
    batch_size = 72
    train_loss_list = []
    val_loss_list = []
    min_list = torch.zeros((180), requires_grad=False)
    max_list = torch.ones((180), requires_grad=False)
    norm_list = torch.stack((min_list, max_list, min_list, max_list), dim = 0)
    
    dir_path = os.path.join(output_root, 'withoutmaxmin1')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_loader, val_loader, test_loader = load_data_bmp(batch_size=batch_size, data_dir=input_root, picture=picture_name, label=label_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = ResNet18(in_channels=n_channels).to(device)
    # model.initialize()
    model = UNet(n_channels=3, n_classes = 4).to(device)
    # model.initialize()
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(
    #     torch.load('/data1/rong/projection-from-OCT/output/havepool_con/checkpoint/ckpt_28_loss_0.00670.pth')['net'])
    
    
    criterion = MaxPoolLoss()
    criterion_2 = PerceptualLoss([2,2,2], [0.6,0.3,0.1], device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    dir_list = os.listdir('/data1/rong/projection-from-OCT/output/havepool/checkpoint')
    for i in dir_list:
        strs = i.split("_")
        # restore_model_path = os.path.join(
        #     dir_path, 'ckpt_'+strs[1]+'_loss_'+strs[3])
        model.load_state_dict(torch.load('/data1/rong/projection-from-OCT/output/havepool/checkpoint/'+'ckpt_'+strs[1]+'_loss_'+strs[3])['net'])
        val(model, criterion, val_loader, device, label_name, val_loss_list, strs[1],train_loss_list)

    loss_list = np.array(val_loss_list)
    index = loss_list.argmin()
    print(train_loss_list[index])
    print('epoch %s is the best model' % (index))

    # print('==> Testing model...')
    # restore_model_path = os.path.join(
    #     dir_path, 'ckpt_%d_loss_%.5f.pth' % (index, loss_list[index]))
    # model.load_state_dict(torch.load(restore_model_path)['net'])
    # test(model, 'test', test_loader, device, flag, task, output_root=output_root)
    # writer.close()


def val(model, criterion,train_loader, device, label_name, val_loss_list, num, train_loss_list):
    model.eval()
    train_loss_b2 = metrics.LossAverage()   
    train_loss_b3 = metrics.LossAverage()  
    train_loss = metrics.LossAverage()
    min_list = torch.zeros((20), requires_grad=False)
    max_list = torch.ones((20), requires_grad=False)
    min_max_list = torch.stack((max_list, min_list, max_list, min_list), dim = 0)
    with torch.no_grad():
        for batch_idx, (or_inputs, inputs, targets) in enumerate(train_loader):
            or_inputs, inputs, targets = or_inputs.to(device), inputs.to(device), targets.to(device)
            predict = model(inputs).to(device)
            i = targets[:,0,:,0].cpu().numpy()
            coor_b2, coor_b3, result_b2, result_b3 = pixel_from_image(or_inputs, predict, 64, 464, device, label_name)  ##change
            min1 ,max1 = torch.min(result_b2[:,0,:,:],2), torch.max(result_b2[:,0,:,:],2)
            resultn_b2 = torch.zeros((inputs.size(0), 3, 1, 464), requires_grad=True).to(device)  
            min2 ,max2 = torch.min(result_b3[:,0,:,:],2), torch.max(result_b3[:,0,:,:],2)
            resultn_b3 = torch.zeros((inputs.size(0), 3, 1, 464), requires_grad=True).to(device)  
            for j in range(len(i)):    
                if min1[0].data[j, 0] < min_max_list[0, np.int(i[j][0])]:
                    min_max_list[0, np.int(i[j][0])] = min1[0].data[j, 0]
                if max1[0].data[j, 0] > min_max_list[1, np.int(i[j][0])]:
                    min_max_list[1, np.int(i[j][0])] = max1[0].data[j, 0] 
                if min2[0].data[j, 0] < min_max_list[2, np.int(i[j][0])]:
                    min_max_list[2, np.int(i[j][0])] = min2[0].data[j, 0]
                if max2[0].data[j, 0] > min_max_list[3, np.int(i[j][0])]:
                    min_max_list[3, np.int(i[j][0])] = max2[0].data[j, 0]      
            for j in range(len(i)):
                resultn_b2[j,:,:,:] = (result_b2[j,:,:,:]-  min_max_list[0, np.int(i[j][0])])/( min_max_list[1, np.int(i[j][0])] -  min_max_list[0, np.int(i[j][0])])   
                resultn_b3[j,:,:,:] = (result_b3[j,:,:,:]-  min_max_list[2, np.int(i[j][0])])/( min_max_list[3, np.int(i[j][0])] -  min_max_list[2, np.int(i[j][0])]) 

            loss_b2 = criterion(resultn_b2, targets[:,:,:,1:465])
            loss_b3 = criterion(resultn_b3, targets[:,:,:,465:929])

            loss = loss_b2 + loss_b3

            train_loss_b2.update(loss_b2.item(),inputs.size(0))
            train_loss_b3.update(loss_b3.item(),inputs.size(0))
            train_loss.update(loss.item(),inputs.size(0))
            val_loss_list.append(train_loss.avg)
            train_loss_list.append(num)
        print(num, train_loss_b2.avg, train_loss_b3.avg, train_loss.avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model')
    parser.add_argument('--input_root',
                        default='/data1/rong/OCT_and_OCTA/',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--picture_name',
                        default='OCT',
                        help='which dataset to choose, OCT or OCTA',
                        type=str)     
    parser.add_argument('--label_name',
                        default='B3',
                        help='which projection map to use',
                        type=str)                    
    parser.add_argument('--num_epoch',
                        default=200,
                        help='num of epochs of training',
                        type=int)

    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    picture_name = args.picture_name
    label_name = args.label_name
    end_epoch = args.num_epoch
    fh = open(input_root+'test.txt', 'r')
    main(input_root, output_root, picture_name, label_name, end_epoch)