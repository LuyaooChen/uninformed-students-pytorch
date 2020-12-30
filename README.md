# Uninformed Students
![result](https://raw.githubusercontent.com/LuyaooChen/uninformed-students-pytorch/main/res.jpg)
## Introduction - 介绍
A simple and incomplete implementation of paper:  
MVTec, [Uninformed Students: Student–Teacher Anomaly Detection with Discriminative Latent Embeddings.](https://ieeexplore.ieee.org/document/9157778/) CVPR, 2020.  
[arXiv:1911.02357](https://arxiv.org/abs/1911.02357)
  
Another implementation repo: https://github.com/denguir/student-teacher-anomaly-detection  

此项目复现主要是本人学习之用，可能存在各种问题。有个朋友创了一个QQ群，欢迎加群讨论：689772351

## Requirements - 依赖
python3  
pytorch~=1.3  
torchvision  
numpy  
opencv-python

## Usage - 用法
### Prepare datasets
imagenet (any image dataset)  
MVTec_AD
### Train a teacher network
choose a `patch_size` from (17, 33 or 65) and  
`python teacher_train.py`
### Train a student network
choose a `patch_size`(the teacher net should have been pretrained), and set `st_id`  
`python student_train.py`
### Evaluate
`python evaluate.py`  
the res.jpg will be saved to the current directory.  

## TODO
metric learning and descriptor compactness in teacher_train.py   
complete evaluate.py  
...

## Reference - 参考
https://github.com/erezposner/Fast_Dense_Feature_Extraction  
https://github.com/denguir/student-teacher-anomaly-detection