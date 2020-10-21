"""
Implementation of Chapt3.2 about students net training in the 'uninformed students' paper.

Author: Luyao Chen
Date: 2020.10
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models import _Teacher, TeacherOrStudent


def loss_fn(student_output: torch.tensor,
            teacher_output: torch.tensor,
            teacher_mean: torch.tensor,
            teacher_std: torch.tensor):
    """
    Loss function for training students to predict teacher's output.

    Args:
    teacher_output: Descriptors out from teacher net, shape: [n*128*imH*imW].
    teacher_mean: Mean of teacher's descriptors, shape: [n*128].
    teacher_std: Standrad deviations of teacher's descriptors, shape: [n*128].
    student_output: Descriptors out from student net, shape: [n*128*imH*imW].
    """
    loss = F.mse_loss(
        student_output,
        (teacher_output - teacher_mean.unsqueeze(-1).unsqueeze(-1)) /
        teacher_std.unsqueeze(-1).unsqueeze(-1)
    )
    return loss


st_id = 0  # student id, start from 0.
# image height and width should be multiples of sL1∗sL2∗sL3...
imH = 512
imW = 512
patch_size = 33
batch_size = 1
epochs = 20
lr = 1e-4
weight_decay = 1e-5
work_dir = 'work_dir/'
device = torch.device('cuda:1')

trans = transforms.Compose([
    transforms.RandomCrop((imH, imW)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# dataset = datasets.ImageNet(
#     '/home/cly/data_disk/imagenet1k/', transform=trans)
dataset = datasets.ImageFolder(
    '/home/cly/data_disk/MVTec_AD/data/wood/train/', transform=trans)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

_teacher = _Teacher(patch_size)
student = TeacherOrStudent(patch_size, _teacher, imH, imW).to(device)

_teacher = _Teacher(patch_size)
checkpoint = torch.load(work_dir + '_teacher' +
                        str(patch_size) + '.pth', torch.device('cpu'))
_teacher.load_state_dict(checkpoint)
teacher = TeacherOrStudent(patch_size, _teacher, imH, imW).to(device)
teacher.eval()

optim = torch.optim.Adam(student.parameters(), lr=lr,
                         weight_decay=weight_decay)

iter_num = 1
for i in range(epochs):
    for data, labels in dataloader:
        data = data.to(device)
        # labels = labels.to(device)
        # with torch.no_grad():
        teacher_output = teacher(data)
        std_mean = torch.std_mean(teacher_output, dim=[2, 3])

        student_output = student(data)
        loss = loss_fn(student_output,
                       teacher_output, std_mean[1], std_mean[0])

        optim.zero_grad()
        loss.backward()
        optim.step()

        if iter_num % 10 == 0:
            print('epoch: {}, iter: {}, loss: {}'.format(
                i + 1, iter_num, loss))
        iter_num += 1
    iter_num = 0

if not os.path.exists(work_dir):
    os.mkdir(work_dir)
print('Saving model to work_dir...')

torch.save(student.state_dict(), work_dir +
           'student' + str(patch_size) + '_' + str(st_id) + '.pth')
