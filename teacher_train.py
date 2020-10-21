"""
Implementation of Chapt3.1 Learning Local Patch Descriptors in the 'uninformed students' paper,
including knowledge distillation, metric learning and descriptor compactness.

Author: Luyao Chen
Date: 2020.10
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from models import _Teacher

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

patch_size = 33
batch_size = 64
lr = 2e-4
weight_decay = 1e-5
epochs = 1
# alpha = 0.9
# temperature = 20
work_dir = 'work_dir/'
device = torch.device('cuda:1')


# def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
#     """
#     Compute the knowledge-distillation (KD) loss given outputs, labels.
#     "Hyperparameters": temperature and alpha
#     """
#     KD_loss = nn.KLDivLoss('batchmean')(F.log_softmax(outputs / T, dim=1),
#                                         F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
#         F.cross_entropy(outputs, labels) * (1. - alpha)

#     return KD_loss


trans = transforms.Compose([
    transforms.RandomResizedCrop(patch_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# dataset = datasets.ImageNet(
#     '/home/cly/data_disk/imagenet1k/', transform=trans)
dataset = datasets.ImageFolder(
    '/home/cly/data_disk/imagenet1k/train/', transform=trans)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

model = _Teacher(patch_size).to(device)
resnet18 = models.resnet18(pretrained=True).to(device)
resnet18.eval()

optim = torch.optim.Adam(model.parameters(), lr=lr,
                         weight_decay=weight_decay)

iter_num = 0
for i in range(epochs):
    for data, labels in dataloader:
        data = data.to(device)
        # labels = labels.to(device)
        output = model(data)
        with torch.no_grad():
            resnet_output = resnet18(data)[:, :512]

        # knowledge distillation loss
        loss_k = F.smooth_l1_loss(output, resnet_output, reduction='sum')
        # loss_k = loss_fn_kd(output, labels, resnet_output, alpha, temperature)
        # metric learning and descriptor compactness are not implemented yet.
        optim.zero_grad()
        loss_k.backward()
        optim.step()

        iter_num += 1
        if iter_num % 10 == 0:
            print('epoch: {}, iter: {}, loss_k: {}'.format(
                i + 1, iter_num, loss_k))
    iter_num = 0

if not os.path.exists(work_dir):
    os.mkdir(work_dir)
print('Saving model to work_dir...')
torch.save(model.state_dict(), work_dir +
           '_teacher' + str(patch_size) + '.pth')
