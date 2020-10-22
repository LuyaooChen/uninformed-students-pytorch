"""
Evaluation for mvtec_ad dataset.
Reference from https://github.com/denguir/student-teacher-anomaly-detection.

Author: Luyao Chen
Date: 2020.10
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import cv2
import numpy as np
from tqdm import tqdm
from models import _Teacher, TeacherOrStudent


def error(student_outputs, teacher_output):
    # n*d*imH*imW
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


def increment_mean_and_var(mu_N, var_N, N, batch):
    '''Increment value of mean and variance based on
       current mean, var and new batch
    '''
    # batch: (batch, h, w)
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
    patch_sizes = [33]  # add more size for multi-scale segmentation
    num_students = 2  # num of studetns per teacher
    imH = 1024  # image height and width should be multiples of sL1∗sL2∗sL3...
    imW = 1024
    batch_size = 1
    work_dir = 'work_dir/'
    device = torch.device('cuda:0')

    N_scale = len(patch_sizes)

    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    trans = transforms.Compose([
        transforms.Resize((imH, imW)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    anomaly_free_dataset = datasets.ImageFolder(
        '/home/cly/data_disk/MVTec_AD/data/wood/train/', transform=trans)
    af_dataloader = DataLoader(anomaly_free_dataset, batch_size=batch_size)
    test_dataset = datasets.ImageFolder(
        '/home/cly/data_disk/MVTec_AD/data/wood/test/', transform=trans)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

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

        # mu_err, var_err = torch.tensor([4.7920308113098145]), torch.tensor([3.410670280456543])
        # mu_var, var_var = torch.tensor([4.074430465698242]), torch.tensor([1.5367100238800049])

        max_err, max_var = torch.zeros((N_scale)), torch.zeros((N_scale))
        mu_err, var_err, N_err = torch.zeros((N_scale)), torch.zeros((N_scale)), 0
        mu_var, var_var, N_var = torch.zeros((N_scale)), torch.zeros((N_scale)), 0
        print('Calibration by anomaly-free data...')
        for data, _ in tqdm(af_dataloader):
            data = data.to(device)
            for i in range(N_scale):
                teacher_output = teachers[i](data)
                student_outputs = []
                for j in range(num_students):
                    student_outputs.append(students[i][j](data))
                e = error(student_outputs, teacher_output)
                v = variance(student_outputs)
                mu_err[i], var_err[i], N_err = increment_mean_and_var(
                    mu_err[i], var_err[i], N_err, e)
                mu_var[i], var_var[i], N_var = increment_mean_and_var(
                    mu_var[i], var_var[i], N_var, v)

                max_err[i] = max(max_err[i], torch.max(e))
                max_var[i] = max(max_var[i], torch.max(v))

        # max_score = 29.9642391204834
        max_score = 0
        for i in range(N_scale):
            print('mu_err:{}, var_err:{}, mu_var:{}, var_var:{}'.format(
                mu_err[i],var_err[i],mu_var[i],var_var[i]
            ))
            max_score += (max_err[i] - mu_err[i]) / torch.sqrt(var_err[i]) + \
                (max_var[i] - mu_var[i]) / torch.sqrt(var_var[i])
        max_score /= N_scale
        print('max_score:{}'.format(max_score))

        for data, _ in test_dataloader:
            ori_imgs = data
            data = data.to(device)
            # anomaly_score = torch.zeros((batch_size, imH, imW)).to(device)
            for i in range(N_scale):
                teacher_output = teachers[i](data)
                student_outputs = []
                for j in range(num_students):
                    student_outputs.append(students[i][j](data))
                e = error(student_outputs, teacher_output)
                v = variance(student_outputs)
                if i == 0:
                    anomaly_score = (e - mu_err[i]) / torch.sqrt(var_err[i]) + \
                        (v - mu_var[i]) / torch.sqrt(var_var[i])
                else:
                    anomaly_score += (e - mu_err[i]) / torch.sqrt(var_err[i]) + \
                        (v - mu_var[i]) / torch.sqrt(var_var[i])

            anomaly_score /= N_scale
            print('max:{:.2f},min:{:.2f},avg:{:.2f}'.format(torch.max(anomaly_score),
                                                torch.min(anomaly_score),
                                                torch.mean(anomaly_score)))

            
            anomaly_score -= torch.min(anomaly_score)
            # anomaly_score /= torch.max(anomaly_score)
            anomaly_score /= max_score
            score_map = anomaly_score.cpu().data.numpy()[0,:,:]
            score_map = np.minimum(score_map,1)
            score_map = cv2.applyColorMap(
                np.uint8(score_map * 255), cv2.COLORMAP_JET)
            # # cv2.imwrite('score.jpg', score_map)
            ori_img = ori_imgs.permute(0, 2, 3, 1).data.numpy()[0, :, :, :]
            for c in range(3):
                ori_img[:, :, c] = ori_img[:, :, c] * std[c] + mean[c]
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            # # cv2.imwrite('ori.jpg', np.uint8(ori_img * 255))
            save_img = np.concatenate((np.uint8(ori_img * 255), score_map), axis=1)
            cv2.imwrite('res.jpg', save_img)
