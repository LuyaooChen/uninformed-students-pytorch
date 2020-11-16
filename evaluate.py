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
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
from models import _Teacher, TeacherOrStudent
from mvtec_dataset import MVTec_AD


def error(student_outputs, teacher_output):
    # n*imH*imW*d
    # s_mean = 0
    # for s_out in student_outputs:
    #     s_mean += s_out
    # s_mean /= len(student_outputs)
    s_mean = torch.mean(student_outputs, dim=1)
    return torch.norm(s_mean - teacher_output, dim=3)


def variance(student_outputs):
    # s_sum = 0
    # for s_out in student_outputs:
    #     s_sum += s_out
    # s_mean = s_sum / len(student_outputs)

    # v = 0
    # for s_out in student_outputs:
    #     v += torch.norm(s_out - s_mean, dim=3)
    # v /= len(student_outputs)
    sse = torch.sum(student_outputs ** 2, dim=4)
    msse = torch.mean(sse, dim=1)
    s_mean = torch.mean(student_outputs, dim=1)
    var = msse - torch.sum(s_mean**2, dim=3)
    return var


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
    patch_sizes = [33]  # add more size for multi-scale segmentation
    num_students = 2  # num of studetns per teacher
    imH = 256  # image height and width should be multiples of sL1∗sL2∗sL3...
    imW = 256
    batch_size = 1
    work_dir = 'work_dir/'
    class_dir = 'leather/'
    # train_dataset_dir = '/home/cly/data_disk/印花布/normal/3/'
    # test_dataset_dir = '/home/cly/data_disk/印花布/瑕疵布/3/'
    train_dataset_dir = '/home/cly/data_disk/MVTec_AD/data/' + class_dir + 'train/'
    test_dataset_dir = '/home/cly/data_disk/MVTec_AD/data/' + class_dir
    device = torch.device('cuda:1')

    N_scale = len(patch_sizes)

    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    trans = transforms.Compose([
        # transforms.RandomCrop((imH, imW)),
        transforms.Resize((imH, imW)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mask_trans = transforms.Compose([
        # transforms.RandomCrop((imH, imW)),
        transforms.Resize((imH, imW), Image.NEAREST),
        transforms.ToTensor(),
    ])
    anomaly_free_dataset = datasets.ImageFolder(
        train_dataset_dir, transform=trans)
    af_dataloader = DataLoader(anomaly_free_dataset, batch_size=batch_size)
    test_dataset = MVTec_AD(test_dataset_dir, transform=trans,
                            mask_transform=mask_trans, phase='test')
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
            # issue #2. must create a new _teacher.
            _teacher = _Teacher(patch_size)
            student = TeacherOrStudent(
                patch_size, _teacher, imH, imW).to(device)
            checkpoint = torch.load(work_dir + class_dir + 'student' +
                                    str(patch_size) + '_' + str(i) +
                                    '.pth', torch.device('cpu'))
            student.load_state_dict(checkpoint)
            student.eval()
            s_t.append(student)
        students.append(s_t)

    with torch.no_grad():
        t_mu, t_var, t_N = [0 for i in range(N_scale)], [0 for i in range(N_scale)], [
            0 for i in range(N_scale)]
        print('Callibrating teacher on train dataset.')
        for data, _ in tqdm(af_dataloader):
            data = data.to(device)
            for i in range(N_scale):
                t_out = teachers[i](data)
                t_mu[i], t_var[i], t_N[i] = increment_mean_and_var(
                    t_mu[i], t_var[i], t_N[i], t_out)

        # mu_err, var_err = torch.tensor([4.7920308113098145]), torch.tensor([3.410670280456543])
        # mu_var, var_var = torch.tensor([4.074430465698242]), torch.tensor([1.5367100238800049])

        max_err, max_var = [0 for i in range(N_scale)], [0 for i in range(N_scale)]
        mu_err, var_err, N_err = [0 for i in range(N_scale)], [0 for i in range(N_scale)], [0 for i in range(N_scale)]
        mu_var, var_var, N_var = [0 for i in range(N_scale)], [0 for i in range(N_scale)], [0 for i in range(N_scale)]
        print('Callibrating scoring parameters on Student dataset.')
        for data, _ in tqdm(af_dataloader):
            data = data.to(device)
            for i in range(N_scale):
                teacher_output = (teachers[i](
                    data) - t_mu[i]) / torch.sqrt(t_var[i])
                student_outputs = []
                for j in range(num_students):
                    student_outputs.append(students[i][j](data))
                student_outputs = torch.stack(student_outputs, dim=1)
                e = error(student_outputs, teacher_output)
                v = variance(student_outputs)
                mu_err[i], var_err[i], N_err[i] = increment_mean_and_var(
                    mu_err[i], var_err[i], N_err[i], e)
                mu_var[i], var_var[i], N_var[i] = increment_mean_and_var(
                    mu_var[i], var_var[i], N_var[i], v)

                max_err[i] = max(max_err[i], torch.max(e))
                max_var[i] = max(max_var[i], torch.max(v))

        # max_score = 29.9642391204834
        max_score = 0
        for i in range(N_scale):
            print('mu_err:{}, var_err:{}, mu_var:{}, var_var:{}'.format(
                mu_err[i], var_err[i], mu_var[i], var_var[i]
            ))
            max_score += (max_err[i] - mu_err[i]) / torch.sqrt(var_err[i]) + \
                (max_var[i] - mu_var[i]) / torch.sqrt(var_var[i])
        max_score /= N_scale
        print('max_score:{}'.format(max_score))

        score_map_list = []
        gt_mask_list = []
        for data, gt_mask, _ in test_dataloader:
            plt_list = []
            ori_imgs = data
            data = data.to(device)
            gt_mask_list.append(gt_mask.data.numpy())
            anomaly_score = 0
            for i in range(N_scale):
                teacher_output = (teachers[i](
                    data) - t_mu[i]) / torch.sqrt(t_var[i])
                plt_list.append(teacher_output)
                student_outputs = []
                for j in range(num_students):
                    student_outputs.append(students[i][j](data))
                    plt_list.append(student_outputs[j])
                student_outputs = torch.stack(student_outputs, dim=1)
                e = error(student_outputs, teacher_output)
                v = variance(student_outputs)
                anomaly_score += (e - mu_err[i]) / torch.sqrt(var_err[i]) + \
                    (v - mu_var[i]) / torch.sqrt(var_var[i])

            anomaly_score /= N_scale
            score_map_list.append(anomaly_score.cpu().detach().numpy())
            # print('max:{:.2f},min:{:.2f},avg:{:.2f}'.format(torch.max(anomaly_score),
            #                                                 torch.min(anomaly_score),
            #                                                 torch.mean(anomaly_score)))

            plt.figure()
            plt.subplot(2, 2, 1)
            plt_img = plt_list[1].cpu().detach().numpy()[0]
            plt_img = np.mean(plt_img, axis=2)
            plt_img = np.expand_dims(plt_img, 2)
            plt.imshow(plt_img, cmap='jet')
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt_img = plt_list[2].cpu().detach().numpy()[0]
            plt_img = np.mean(plt_img, axis=2)
            plt_img = np.expand_dims(plt_img, 2)
            plt.imshow(plt_img, cmap='jet')
            plt.colorbar()
            # plt.subplot(2, 2, 3)
            # plt_img = plt_list[3].cpu().detach().numpy()[0]
            # plt_img = np.mean(plt_img, axis=2)
            # plt_img = np.expand_dims(plt_img, 2)
            # plt.imshow(plt_img, cmap='jet')
            # plt.colorbar()
            plt.subplot(2, 2, 4)
            plt_img = plt_list[0].cpu().detach().numpy()[0]
            plt_img = np.mean(plt_img, axis=2)
            plt_img = np.expand_dims(plt_img, 2)
            plt.imshow(plt_img, cmap='jet')
            plt.colorbar()
            plt.savefig('cmp.png')
            plt.close()

            px = 118
            py = 132
            plt.figure(figsize=(6, 3))
            plt_vec = plt_list[1].cpu().detach().numpy()[0, px, py]
            plt_vec -= plt_list[2].cpu().detach().numpy()[0, px, py]
            plt.plot(plt_vec, label='s1')
            # plt_vec = plt_list[2].cpu().detach().numpy()[0, px, py]
            # plt.plot(plt_vec, label='s2')
            # plt_vec = plt_list[3].cpu().detach().numpy()[0, px, py]
            # plt.plot(plt_vec, label='s3')
            # plt_vec = plt_list[0].cpu().detach().numpy()[0, px, py]
            # plt.plot(plt_vec, label='t')
            plt.legend()
            plt.savefig('vec.png')
            plt.close()

            anomaly_score -= torch.min(anomaly_score)
            # anomaly_score /= torch.max(anomaly_score)
            # anomaly_score /= max_score
            anomaly_score /= 30
            score_map = anomaly_score.cpu().detach().numpy()[0, :, :]
            score_map = np.minimum(score_map, 1)
            score_map = cv2.applyColorMap(
                np.uint8(score_map * 255), cv2.COLORMAP_JET)
            # # cv2.imwrite('score.jpg', score_map)
            ori_img = ori_imgs.permute(0, 2, 3, 1).detach().numpy()[0, :, :, :]
            for c in range(3):
                ori_img[:, :, c] = ori_img[:, :, c] * std[c] + mean[c]
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            # # cv2.imwrite('ori.jpg', np.uint8(ori_img * 255))
            save_img = np.concatenate(
                (np.uint8(ori_img * 255), score_map), axis=1)
            cv2.imwrite('res.jpg', save_img)

        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        per_pixel_rocauc = roc_auc_score(
            flatten_gt_mask_list, flatten_score_map_list)
        print('pixel ROCAUC:{}'.format(per_pixel_rocauc))
