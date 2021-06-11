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
from utils import metrics
from collections import OrderedDict
import argparse
from models.projection import pixel_from_image

def main(input_root, output_root, picture_name, label_name, end_epoch):
    n_channels = 3
    start_epoch = 0
    lr = 0.001
    val_loss_list = []
    dir_path = os.path.join(output_root, 'checkpoints_b21')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_loader, val_loader, test_loader = load_data_bmp(batch_size=4, data_dir=input_root, picture=picture_name, label=label_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels).to(device)
    model = torch.nn.DataParallel(model)
    criterion = MaxPoolLoss(10,400)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, label_name)
        val(model, criterion, val_loader, device, val_loss_list, dir_path, epoch, label_name)

    loss_list = np.array(val_loss_list)
    index = loss_list.argmin()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_loss_%.5f.pth' % (index, loss_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    # test(model, 'test', test_loader, device, flag, task, output_root=output_root)

def train(model, optimizer, criterion, train_loader, device, label_name):
    model.train()
    train_loss = metrics.LossAverage()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),total = len(train_loader)):
        optimizer.zero_grad()
        predict = model(inputs.to(device))
        targets = targets.to(device)
        predict = predict.to(device)
        pixel_coord, result = pixel_from_image(inputs.to(device), predict, 10, 400, device, label_name)
        print(pixel_coord[0,0,:]*320+320)
        loss = criterion(result, targets)
        loss.backward()
        optimizer.step()
        # for name, parms in model.named_parameters():	
	    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		#  ' -->grad_value:',parms.grad)
        train_loss.update(loss.item(),inputs.size(0))
    print(train_loss.avg)

def val(model, criterion, val_loader, device, val_loss_list, dir_path, epoch, label_name):
    model.eval()
    train_loss = metrics.LossAverage()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            targets = targets.to(device)
            predict = model(inputs.to(device)).to(device)
            # targets, outputs = targets.cpu().numpy(), outputs.cpu().numpy()
            pixel_coord, result = pixel_from_image(inputs.to(device), predict, 10, 400, device, label_name)
            loss = criterion(result, targets)
            # loss = criterion(inputs.to(device), predict, targets, device)
            train_loss.update(loss.item(),inputs.size(0))
        val_loss_list.append(train_loss.avg)

    state = {
        'net': model.state_dict(),
        'loss': train_loss.avg,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_loss_%.5f.pth' % (epoch, train_loss.avg))
    torch.save(state, path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model')
    parser.add_argument('--input_root',
                        default='/media/rong/file/OCT and OCTA/',
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
                        default='B2',
                        help='which projection map to use',
                        type=str)                    
    parser.add_argument('--num_epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)

    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    picture_name = args.picture_name
    label_name = args.label_name
    end_epoch = args.num_epoch
    main(input_root, output_root, picture_name, label_name, end_epoch)