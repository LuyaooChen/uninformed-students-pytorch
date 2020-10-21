"""
Evaluation for mvtec_ad dataset.

Author: Luyao Chen
Date: 2020.10
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import cv2
import numpy as np
from models import _Teacher, TeacherOrStudent


def error(student_outputs, teacher_output):
    t_std_mean = torch.std_mean(teacher_output, dim=[2, 3])

    for i, s_out in enumerate(student_outputs):
        if i == 0:
            e = torch.norm(s_out - (teacher_output - t_std_mean[1].unsqueeze(-1).unsqueeze(-1)) /
                           t_std_mean[0].unsqueeze(-1).unsqueeze(-1), dim=1)
        else:
            e += torch.norm(s_out - (teacher_output - t_std_mean[1].unsqueeze(-1).unsqueeze(-1)) /
                            t_std_mean[0].unsqueeze(-1).unsqueeze(-1), dim=1)
    # n*imH*imW
    e /= len(student_outputs)
    return e


def variance(student_outputs):
    for i, s_out in enumerate(student_outputs):
        if i == 0:
            s_sum = s_out
        else:
            s_sum += s_out
    s_mean = s_sum / len(student_outputs)
    for i, s_out in enumerate(student_outputs):
        if i == 0:
            # v = torch.norm(s_out, dim=1) - torch.norm(s_mean, dim=1)
            v = torch.norm(s_out - s_mean, dim=1)
        else:
            # v += torch.norm(s_out, dim=1) - torch.norm(s_mean, dim=1)
            v += torch.norm(s_out - s_mean, dim=1)
    v /= len(student_outputs)
    return v


patch_sizes = [33]  # add more size for multi-scale segmentation
num_students = 2  # num of studetns per teacher
imH = 1024  # image height and width should be multiples of sL1∗sL2∗sL3...
imW = 1024
batch_size = 1
work_dir = 'work_dir/'
device = torch.device('cuda:1')

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

trans = transforms.Compose([
    transforms.Resize((imH, imW)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# dataset = datasets.ImageNet(
#     '/home/cly/data_disk/imagenet1k/', transform=trans)
dataset = datasets.ImageFolder(
    '/home/cly/data_disk/MVTec_AD/data/wood/test/', transform=trans)
dataloader = DataLoader(dataset, batch_size=batch_size)

teachers = []
students = []
for patch_size in patch_sizes:
    _teacher = _Teacher(patch_size)
    checkpoint = torch.load(work_dir + '_teacher' +
                            str(patch_size) + '.pth', torch.device('cpu'))
    _teacher.load_state_dict(checkpoint)
    teacher = TeacherOrStudent(patch_size, _teacher, imH, imW).to(device)
    teacher.eval()
    teachers.append(teacher)

    s_t = []
    for i in range(num_students):
        student = TeacherOrStudent(patch_size, _teacher, imH, imW).to(device)
        checkpoint = torch.load(work_dir + 'student' +
                                str(patch_size) + '_' + str(i) +
                                '.pth', torch.device('cpu'))
        student.load_state_dict(checkpoint)
        student.eval()
        s_t.append(student)
    students.append(s_t)

with torch.no_grad():
    for data, labels in dataloader:
        ori_imgs = data
        data = data.to(device)
        # anomaly_score = torch.zeros((batch_size, imH, imW)).to(device)
        for i in range(len(patch_sizes)):
            teacher_output = teachers[i](data)
            student_outputs = []
            for j in range(num_students):
                student_outputs.append(students[i][j](data))
            e = error(student_outputs, teacher_output)
            v = variance(student_outputs)
            e_std_mean = torch.std_mean(e, dim=[1, 2])
            v_std_mean = torch.std_mean(v, dim=[1, 2])
            if i == 0:
                anomaly_score = (e - e_std_mean[1].unsqueeze(-1).unsqueeze(-1)) / \
                    e_std_mean[0].unsqueeze(-1).unsqueeze(-1) + \
                    (v - v_std_mean[1].unsqueeze(-1).unsqueeze(-1)) / \
                    v_std_mean[0].unsqueeze(-1).unsqueeze(-1)
            else:
                anomaly_score += (e - e_std_mean[1].unsqueeze(-1).unsqueeze(-1)) / \
                    e_std_mean[0].unsqueeze(-1).unsqueeze(-1) + \
                    (v - v_std_mean[1].unsqueeze(-1).unsqueeze(-1)) / \
                    v_std_mean[0].unsqueeze(-1).unsqueeze(-1)

        anomaly_score /= len(patch_sizes)

        anomaly_score -= torch.min(anomaly_score)
        anomaly_score /= torch.max(anomaly_score)
        score_map = anomaly_score.cpu().data.numpy()[0, :, :]
        score_map = cv2.applyColorMap(
            np.uint8(score_map * 255), cv2.COLORMAP_JET)
        # cv2.imwrite('score.jpg', score_map)
        ori_img = ori_imgs.permute(0, 2, 3, 1).data.numpy()[0, :, :, :]
        for c in range(3):
            ori_img[:, :, c] = ori_img[:, :, c] * std[c] + mean[c]
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('ori.jpg', np.uint8(ori_img * 255))
        save_img = np.concatenate((np.uint8(ori_img * 255), score_map), axis=1)
        cv2.imwrite('res.jpg', save_img)
