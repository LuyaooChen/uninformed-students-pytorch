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
from tqdm import tqdm
from models import _Teacher, TeacherOrStudent


def increment_mean_and_var(mu_N, var_N, N, batch):
    '''Increment value of mean and variance based on
       current mean, var and new batch
    '''
    # batch: (batch, h, w, vector)
    B = batch.size()[0]  # batch size
    # we want a descriptor vector -> mean over batch and pixels
    mu_B = torch.mean(batch, dim=[0, 1, 2])
    S_B = B * torch.var(batch, dim=[0, 1, 2], unbiased=False)
    S_N = N * var_N
    mu_NB = N / (N + B) * mu_N + B / (N + B) * mu_B
    S_NB = S_N + S_B + B * mu_B**2 + N * mu_N**2 - (N + B) * mu_NB**2
    var_NB = S_NB / (N + B)
    return mu_NB, var_NB, N + B

if __name__ == "__main__":
    
    st_id = 0  # student id, start from 0.
    # image height and width should be multiples of sL1∗sL2∗sL3...
    imH = 512
    imW = 512
    patch_size = 17
    batch_size = 1
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-5
    work_dir = 'work_dir/'
    class_dir = 'leather/'
    dataset_dir = '/home/cly/data_disk/MVTec_AD/data/' + class_dir + 'train/'
    # dataset_dir = '/home/cly/data_disk/印花布/normal/3/'
    device = torch.device('cuda:1')

    trans = transforms.Compose([
        transforms.Resize((imH, imW)),
        # transforms.RandomCrop((imH, imW)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(dataset_dir, transform=trans)
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

    with torch.no_grad():
        t_mu, t_var, N = 0, 0, 0
        for data, _ in tqdm(dataloader):
            data = data.to(device)
            t_out = teacher(data)
            t_mu, t_var, N = increment_mean_and_var(t_mu, t_var, N, t_out)

    optim = torch.optim.Adam(student.parameters(), lr=lr,
                            weight_decay=weight_decay)

    iter_num = 1
    for i in range(epochs):
        for data, labels in dataloader:
            data = data.to(device)
            # labels = labels.to(device)
            with torch.no_grad():
                teacher_output = (teacher(data) - t_mu) / torch.sqrt(t_var)

            student_output = student(data)
            loss = F.mse_loss(student_output, teacher_output)

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
    if not os.path.exists(work_dir + class_dir):
        os.mkdir(work_dir + class_dir)
    print('Saving model to work_dir...')

    torch.save(student.state_dict(), work_dir + class_dir +
            'student' + str(patch_size) + '_' + str(st_id) + '.pth')
