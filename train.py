from models import ResNet18, ResNet50
from dataset10001 import load_data_bmp
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from loss import MaxPoolLoss
from tqdm import trange
from tqdm import tqdm
import metrics
from collections import OrderedDict
from torch.autograd import Variable


def main(end_epoch, input_root, output_root):
    n_channels = 3
    start_epoch = 0
    lr = 0.001
    val_auc_list = []
    dir_path = os.path.join(output_root, 'checkpoints')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_loader, val_loader = load_data_bmp(batch_size=4, data_dir=input_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels).to(device)
    model = torch.nn.DataParallel(model)
    criterion = MaxPoolLoss(1,400)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device)
        # val(model, val_loader, device, val_auc_list, dir_path, epoch)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    # model.load_state_dict(torch.load(restore_model_path)['net'])

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = metrics.LossAverage()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),total = len(train_loader)):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        targets = targets.to(device)
        loss = criterion(inputs.to(device), outputs, targets, device)
        print(outputs)
        # loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    #     # for name, parms in model.named_parameters():	
	#     #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
	# 	#  ' -->grad_value:',parms.grad)
        train_loss.update(loss.item(),inputs.size(0))
    print(train_loss.avg)

def val(model, val_loader, device, val_auc_list, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(torch.float32).to(device)).to(torch.float32)
            targets = targets.to(torch.float32).to(device)


            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)
if __name__ == '__main__':
    input_root='/media/rong/file/OCT and OCTA/'
    main(100, input_root, output_root='./output')
    # print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    # model.train()
    # for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
    #     data, target = data.float(), target.long()
    #     target = common.to_one_hot_3d(target,n_labels)
    #     data, target = data.to(device), target.to(device)
    #     optimizer.zero_grad()

    #     output = model(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()

    # if n_labels == 2:
    #     return OrderedDict({'Train Loss': train_loss.avg, 'Train dice0': train_dice.avg[0],
    #                    'Train dice1': train_dice.avg[1]})
    # else:
    #     return OrderedDict({'Train Loss': train_loss.avg, 'Train dice0': train_dice.avg[0],
    #                    'Train dice1': train_dice.avg[1],'Train dice2': train_dice.avg[2]})




# image  = cv2.imread('/media/rong/file/OCT and OCTA/OCTA_6M_OCTA/10001/1.bmp')
# pic = cv2.imread('/media/rong/file/OCT and OCTA/OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/10001.bmp')
# transform = transforms.ToTensor()
# img = transform(image)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ResNet18(in_channels=n_channels).to(device)
# Y = torch.rand((2, 3, 640, 400))
# result = model(Y.to(device))
# print(result.shape)

# def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
#     net = net.to(device)
#     print("training on ", device)
#     loss = torch.nn.CrossEntropyLoss()
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
#         for X, y in train_iter:
#             X = X.to(device)
#             y = y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
#               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))