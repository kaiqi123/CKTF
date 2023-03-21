# Contrastive Knowledge Transfer Framework (CKTF)

This repo implements the paper published in **ICASSP 2023 (Oral)**:

**A Contrastive Knowledge Transfer Framework for Model Compression and Transfer Learning** (termed CKTF)

The link of the paper is: https://arxiv.org/pdf/2303.07599.pdf.

The overall workflow of CKTF is as follows:

![Workflow of CKTF](https://github.com/kaiqi123/CKTF/blob/680d36c14375e3b0b6469cd85da052dc09698349/CKTF_pattern.png)

## Installation
We implemented CKTF on PyTorch version 1.9.0 and CUDA 11.2, and conducted experiments on 4 Nvidia RTX 2080 Ti GPUs.

## Running

1. For the model compression results in Table 1 of the paper:

    (1) Run CKTF and 13 other KD methods on CIFAR-100, follow the commands in `scripts/run_cifar100_distill.sh`. 

    (2) Run CKTF and 13 other KD methods on Tiny-ImageNet, follow the commands in `scripts/run_tiny-imagenet_distill.sh`. 

2. For model compression results in Table 2 of the paper, follow the commands in `scripts/run_2kds.sh`. 

3. For the transfer learning results in Figure 2 of the paper (Transfer learning from tiny-imagenet to stl10), follow the the commands in `scripts/run_transfer_from_tinyimagenet_to_stl10.sh`. 

4. Train the teacher model from scratch, follow the commands in `scripts/run_vanilla_models.sh`. 

## Results

1. Top-1 test accuracy (\%) on CIFAR-100 for model compressin.

|  Teacher /Student  | WRN-40-2 /WRN-16-2 | WRN-40-2 /WRN-40-1 | ResNet-56 /ResNet-20 | ResNet-110 /ResNet-20 | ResNet-110 /ResNet-32 | ResNet-32X4 /ResNet-8 X4 | VGG-13 /VGG-8 |
|:-----------------:|:-----------------:|:-----------------:|:-------------------:|:--------------------:|:--------------------:|:-----------------------:|:------------:|
| Compression Ratio | 3.21              | 3.96              | 3.1                 | 6.24                 | 3.67                 | 6.03                    | 2.39         |
| Teacher           | 75.61             | 75.61             | 72.34               | 74.31                | 74.31                | 79.42                   | 74.64        |
| Student (w/o KD)  | 73.26             | 73.54             | 69.06               | 69.06                | 71.14                | 72.5                    | 70.36        |
| KD                | 74.92             | 73.54             | 70.66               | 70.67                | 73.08                | 73.33                   | 72.98        |
| FitNet            | 73.58             | 72.24             | 69.21               | 68.99                | 71.06                | 73.5                    | 71.02        |
| AT                | 74.08             | 72.77             | 70.55               | 70.22                | 72.31                | 73.44                   | 71.43        |
| SP                | 73.83             | 72.43             | 69.67               | 70.04                | 72.69                | 72.94                   | 72.68        |
| CC                | 73.56             | 72.21             | 69.63               | 69.48                | 71.48                | 72.97                   | 70.71        |
| VID               | 74.11             | 73.3              | 70.38               | 70.16                | 72.61                | 73.09                   | 71.23        |
| RKD               | 73.35             | 72.22             | 69.61               | 69.25                | 71.82                | 71.9                    | 71.48        |
| PKT               | 74.54             | 73.45             | 70.34               | 70.25                | 72.61                | 73.64                   | 72.88        |
| AB                | 72.5              | 72.38             | 69.47               | 69.53                | 70.98                | 73.17                   | 70.94        |
| FT                | 73.25             | 71.59             | 69.84               | 70.22                | 72.37                | 72.86                   | 70.58        |
| FSP               | 72.91             | N/A               | 69.95               | 70.11                | 71.89                | 72.62                   | 70.23        |
| NST               | 73.68             | 72.24             | 69.6                | 69.53                | 71.96                | 73.3                    | 71.53        |
| CRD               | 75.48             | 74.14             | 71.16               | 71.46                | 73.48                | 75.51                   | 73.94        |
| **CKTF**          | **75.85**         | **74.49**         | **71.2**            | **71.8**             | **73.84**            | **75.74**               | **74.31**    |
| CRD+KD            | 75.64             | 74.38             | 71.63               | 71.56                | 73.75                | 75.46                   | 74.29        |
| **CKTF+KD**       | **75.89**         | **74.94**         | **71.86**           | **71.66**            | **74.07**            | **75.97**               | **74.55**    |


2. Top-1 test accuracy (\%) on Tiny-ImageNet for model compressin.

| Teacher /Student  | VGG-19 /VGG-8 | VGG-16 /VGG-11 | ResNet-34 /ResNet-10 | ResNet-50 /ResNet-10 |
|-------------------|---------------|----------------|----------------------|----------------------|
| Compression Ratio | 5.01          | 1.59           | 4.28                 | 4.78                 |
| Teacher           | 61.62         | 61.35          | 65.38                | 65.34                |
| Student (w/o KD)  | 54.61         | 58.6           | 58.01                | 58.01                |
| KD                | 55.55         | 62.51          | 58.92                | 58.63                |
| FitNet            | 55.24         | 59.08          | 58.22                | 57.76                |
| AT                | 53.55         | 61.4           | 59.16                | 58.92                |
| SP                | 55.09         | 61.61          | 55.91                | 57.17                |
| CC                | 54.87         | 58.34          | 57.18                | 57.36                |
| VID               | 54.94         | 60.07          | 58.53                | 57.65                |
| RKD               | 54.13         | 59.96          | 57.35                | 57.05                |
| PKT               | 55.35         | 60.46          | 58.41                | 58.66                |
| AB                | 50.31         | 55.65          | 57.22                | 58.05                |
| FT                | 53.65         | 58.84          | 56.22                | 56.48                |
| FSP               | N/A           | N/A            | N/A                  | N/A                  |
| NST               | 51.08         | 58.47          | 59.23                | 47.83                |
| CRD               | 56.99         | 62.04          | 60.02                | 59.31                |
| **CKTF**          | **57.57**     | **63.01**      | **60.39**            | **59.42**            |
| CRD+KD            | 58.09         | 63.66          | 61.99                | 61.26                |
| **CKTF+KD**       | **58.76**     | **63.97**      | **62.31**            | **61.51**            |


3. Top-1 test accuracy (%) of KD, CRD, and CKTF on STL-10 when transferring knowledge from Tiny-ImageNet.

(1) T:VGG-19/S:VGG-19

(2) T:VGG-19/S:VGG-8

(3) T:ResNet-18/S:ResNet-18



## Citation

If you think this repo is helpful for your research, please consider citing the paper:
```
@article{zhao2023contrastive,
  title={A Contrastive Knowledge Transfer Framework for Model Compression and Transfer Learning},
  author={Zhao, Kaiqi and Chen, Yitao and Zhao, Ming},
  journal={arXiv preprint arXiv:2303.07599},
  year={2023}
}
```

## Reference

Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive Representation Distillation." International Conference on Learning Representations.

