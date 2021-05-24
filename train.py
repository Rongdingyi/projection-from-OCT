from models import ResNet18, ResNet50
from dataset import load_data_bmp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

n_channels = 3
n_classes = 640
image  = cv2.imread('/media/rong/file/OCT and OCTA/OCTA_6M_OCTA/10001/1.bmp')
pic = cv2.imread('/media/rong/file/OCT and OCTA/OCTA-500_ground_truth/OCTA-500/OCTA_6M/Projection Maps/OCTA(ILM_OPL)/10001.bmp')
transform = transforms.ToTensor()
img = transform(image)

# for picture in load_data_bmp(10,data_dir = '/media/rong/file/OCT and OCTA/OCTA_6M_OCTA/10001'):
#     print(picture.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
Y = torch.rand((2, 3, 640, 400))
result = model(Y.to(device))
# result = result.view(-1,25)
# result = result.cpu().detach().numpy() 
print(result.shape)

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