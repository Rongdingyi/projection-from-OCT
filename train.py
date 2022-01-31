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
from models.projection import projections
from tensorboardX import SummaryWriter
import numpy as np

def main(input_root, output_root, checkpoint_path, end_epoch):
    writer = SummaryWriter()
    lr = 0.0001
    batch_size = 72
    min_list = torch.zeros((100), requires_grad=False)
    max_list = torch.ones((100), requires_grad=False)
    norm_list = torch.stack((min_list, max_list, min_list, max_list), dim = 0)
    val_loss_list = []
    
    dir_path = os.path.join(output_root, 'overfitnew_b2')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_loader, val_loader, test_loader = load_data_bmp(batch_size=batch_size, data_dir=input_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes = 4).to(device)
    model.initialize()
    model = torch.nn.DataParallel(model)
    if checkpoint_path:
        model.load_state_dict(
            torch.load(checkpoint_path)['net'])
    
    criterion = MaxPoolLoss()
    criterion_pl = PerceptualLoss([2,2,2], [0.6,0.3,0.1], device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in trange(0, end_epoch):
        min_max_list = torch.stack((max_list, min_list, max_list, min_list), dim = 0)
        train(model, optimizer, criterion, criterion_pl, train_loader, device, norm_list, min_max_list, dir_path, epoch)
        norm_list = min_max_list        
        scheduler.step()
        val(model, criterion, val_loader, device, val_loss_list)

    loss_list = np.array(val_loss_list)
    index = loss_list.argmin()
    print('epoch %s is the best model' % (index))

    # print('==> Testing model...')
    # restore_model_path = os.path.join(
    #     dir_path, 'ckpt_%d_loss_%.5f.pth' % (index, loss_list[index]))
    # model.load_state_dict(torch.load(restore_model_path)['net'])
    # test(model, 'test', test_loader, device, flag, task, output_root=output_root)
    # writer.close()

def train(model, optimizer, criterion, criterion_pl, train_loader, device, norm_list, min_max_list, dir_path, epoch):
    model.train()
    train_loss = metrics.LossAverage()
    for batch_idx, (or_inputs, inputs, targets) in tqdm(enumerate(train_loader),total = len(train_loader)):
        optimizer.zero_grad()
        or_inputs, inputs, targets = or_inputs.to(device), inputs.to(device), targets.to(device)
        predict = model(inputs).to(device)
        norm_list = norm_list.to(device).detach()
        i = targets[:,0,:,0].cpu().numpy()
        coor_b2, coor_b3, result_b2, result_b3 = projections(or_inputs, predict, 64, 464, device)
        min1 ,max1 = torch.min(result_b2[:,0,:,:],2), torch.max(result_b2[:,0,:,:],2)
        resultn_b2 = torch.zeros((inputs.size(0), 3, 1, 464), requires_grad=True).to(device)  
        min2 ,max2 = torch.min(result_b3[:,0,:,:],2), torch.max(result_b3[:,0,:,:],2)
        resultn_b3 = torch.zeros((inputs.size(0), 3, 1, 464), requires_grad=True).to(device)  
        for j in range(len(i)):    
            resultn_b2[j,:,:,:] = (result_b2[j,:,:,:]- norm_list[0, np.int(i[j][0])])/(norm_list[1, np.int(i[j][0])] - norm_list[0, np.int(i[j][0])]) 
            if min1[0].data[j, 0] < min_max_list[0, np.int(i[j][0])]:
                min_max_list[0, np.int(i[j][0])] = min1[0].data[j, 0]
            if max1[0].data[j, 0] > min_max_list[1, np.int(i[j][0])]:
                min_max_list[1, np.int(i[j][0])] = max1[0].data[j, 0] 
            resultn_b3[j,:,:,:] = (result_b3[j,:,:,:]- norm_list[2, np.int(i[j][0])])/(norm_list[3, np.int(i[j][0])] - norm_list[2, np.int(i[j][0])]) 
            if min2[0].data[j, 0] < min_max_list[2, np.int(i[j][0])]:
                min_max_list[2, np.int(i[j][0])] = min2[0].data[j, 0]
            if max2[0].data[j, 0] > min_max_list[3, np.int(i[j][0])]:
                min_max_list[3, np.int(i[j][0])] = max2[0].data[j, 0]        
        loss1 = criterion_pl(resultn_b2.view(-1, 3, 16,29), targets[:,:,:,1:465].view(-1, 3, 16,29))
        loss2 = criterion_pl(resultn_b3.view(-1, 3, 16,29), targets[:,:,:,465:929].view(-1, 3, 16,29))
        loss_b2 = criterion(resultn_b2, targets[:,:,:,1:465])
        loss_b3 = criterion(resultn_b3, targets[:,:,:,465:929])
        loss = loss_b2+0.25*loss1+0.5*(loss_b3+0.25*loss2)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(),inputs.size(0))
    state = {
        'net': model.state_dict(),
        'loss': train_loss.avg,
        'epoch': epoch,
    }
    save_path = dir_path +'/checkpoint'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = os.path.join(save_path, 'ckpt_%d_loss_%.5f.pth' % (epoch, train_loss.avg))
    torch.save(state, path)


def val(model, criterion,train_loader, device, val_loss_list):
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
            coor_b2, coor_b3, result_b2, result_b3 = projections(or_inputs, predict, 64, 464, device)  ##change
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
            val_loss_list.append(train_loss)
        print(train_loss_b2.avg, train_loss_b3.avg)

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
                        default='None',
                        help='use checkpoint or not',
                        type=str)                   
    parser.add_argument('--end_epoch',
                        default=200,
                        help='num of epochs of training',
                        type=int)

    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    checkpoint_path = args.checkpoint_path
    end_epoch = args.end_epoch
    main(input_root, output_root, checkpoint_path, end_epoch)