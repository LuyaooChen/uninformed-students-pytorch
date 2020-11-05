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


def distillation_loss(output, target):
    # dim: (batch, vector)
    err = torch.norm(output - target, dim=1)**2
    loss = torch.mean(err)
    return loss


def compactness_loss(output):
    # dim: (batch, vector)
    _, n = output.size()
    avg = torch.mean(output, axis=1)
    std = torch.std(output, axis=1)
    zt = output.T - avg
    zt /= std
    corr = torch.matmul(zt.T, zt) / (n - 1)
    loss = torch.sum(torch.triu(corr, diagonal=1)**2)
    return loss


if __name__ == "__main__":
    patch_size = 65
    batch_size = 64
    lr = 2e-4
    weight_decay = 1e-5
    epochs = 2
    # alpha = 0.9
    # temperature = 20
    work_dir = 'work_dir/'
    device = torch.device('cuda:1')

    trans = transforms.Compose([
        transforms.RandomResizedCrop(patch_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(
        '/home/cly/data_disk/imagenet1k/train/', transform=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)

    model = _Teacher(patch_size).to(device)
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-1]).to(device)
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
                resnet_output = resnet18(data).view(-1, 512)

            # knowledge distillation loss
            # loss_k = F.smooth_l1_loss(output, resnet_output, reduction='sum')
            loss_k = distillation_loss(output, resnet_output)
            # metric learning is not implemented yet.
            loss_c = compactness_loss(output)
            loss = loss_k + loss_c
            optim.zero_grad()
            loss.backward()
            optim.step()

            iter_num += 1
            if iter_num % 10 == 0:
                print('epoch:{}, iter:{}, loss_k:{:.3f}, loss_c:{:.3f}, loss:{:.3f}'.format(
                    i + 1, iter_num, loss_k, loss_c, loss))
        iter_num = 0

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    print('Saving model to work_dir...')
    torch.save(model.state_dict(), work_dir +
               '_teacher' + str(patch_size) + '.pth')
