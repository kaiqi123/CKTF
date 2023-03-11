# Contrastive Knowledge Transfer Framework (CKTF)

This repo implements the ICASSP 2023 paper:

"A Contrastive Knowledge Transfer Framework for Model Compression and Transfer Learning" (termed CKTF). 


## Installation
We implemented CKTF on PyTorch version 1.9.0 and CUDA 11.2, and conducted experiments on 4 Nvidia RTX 2080 Ti GPUs.

## Running

1. For the model compression results in Table 1 of the paper:

    (1) Run CKTF and 13 other KD methods on CIFAR-100, follow the commands in `scripts/run_cifar100_distill.sh`. 

    (2) Run CKTF and 13 other KD methods on Tiny-ImageNet, follow the commands in `scripts/run_tiny-imagenet_distill.sh`. 

2. For model compression results in Table 2 of the paper, follow the commands in `scripts/run_2kds.sh`. 

3. For the transfer learning results in Figure 2 of the paper (Transfer learning from tiny-imagenet to stl10), follow the the commands in `scripts/run_transfer_from_tinyimagenet_to_stl10.sh`. 

4. Train the teacher model from scratch, follow the commands in `scripts/run_vanilla_models.sh`. 


## Citation

If you find this repo useful for your research, please consider citing the paper

## Reference

Tian, Yonglong, Dilip Krishnan, and Phillip Isola. "Contrastive Representation Distillation." International Conference on Learning Representations.
