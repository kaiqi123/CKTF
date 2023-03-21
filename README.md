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

