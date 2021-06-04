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
from torch.autograd import Variable
from models.projection import pixel_from_image

def main(end_epoch, input_root, output_root):
    n_channels = 3
    start_epoch = 0
    lr = 0.001
    val_loss_list = []
    dir_path = os.path.join(output_root, 'checkpoints')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_loader, val_loader = load_data_bmp(batch_size=4, data_dir=input_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels).to(device)
    model = torch.nn.DataParallel(model)
    criterion = MaxPoolLoss(10,400)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device)
        val(model, criterion, val_loader, device, val_loss_list, dir_path, epoch)

    loss_list = np.array(val_loss_list)
    index = loss_list.argmin()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_loss_%.5f.pth' % (index, loss_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = metrics.LossAverage()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),total = len(train_loader)):
        optimizer.zero_grad()
        predict = model(inputs.to(device))
        targets = targets.to(device)
        predict = predict.to(device)
        loss = criterion(inputs.to(device), predict, targets, device)
        loss.backward()
        optimizer.step()
        # for name, parms in model.named_parameters():	
	    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		#  ' -->grad_value:',parms.grad)
        train_loss.update(loss.item(),inputs.size(0))
    print(train_loss.avg)

def val(model, criterion, val_loader, device, val_loss_list, dir_path, epoch):
    model.eval()
    train_loss = metrics.LossAverage()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            targets = targets.to(device)
            predict = model(inputs.to(device)).to(device)
            # targets, outputs = targets.cpu().numpy(), outputs.cpu().numpy()
            loss = criterion(inputs.to(device), predict, targets, device)
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
    input_root='/media/rong/file/OCT and OCTA/'
    main(100, input_root, output_root='./output')