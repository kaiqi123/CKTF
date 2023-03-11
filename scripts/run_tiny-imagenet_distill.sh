# =======================================================
# vgg19/vgg8
# batch_size: default 64
# =======================================================





# ==============================================> vgg19/vgg8, without KD (alpha=0.0)<============================================
# kd
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill kd -r 0.1 -a 0.9 -b 0.0 \
--trial 1

# FitNet
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill hint -r 1.0 -a 0 -b 100 \
--trial 1

# AT
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill attention -r 1.0 -a 0 -b 1000 \
--trial 1

# SP
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill similarity -r 1.0 -a 0 -b 3000 \
--trial 1


# CC
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill correlation -r 1.0 -a 0 -b 0.02 \
--trial 1

# VID
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill vid -r 1.0 -a 0 -b 1 \
--trial 1

# RKD
CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill rkd -r 1.0 -a 0 -b 1 \
--trial 1

# PKT
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill pkt -r 1.0 -a 0 -b 30000 \
--trial 1


# AB
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill abound -r 1.0 -a 0 -b 1 \
--trial 1

# FT
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill factor -r 1.0 -a 0 -b 200 \
--trial 1

# FSP
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19_2relus' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill fsp -r 1.0 -a 0 -b 50 \
--trial 1


# NST, bs is set to 32, due to out of memory
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19_2relus' \
--batch_size 32 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill nst -r 1.0 -a 0 -b 50 \
--trial 1


# CRD
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19_2relus' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill crd -r 1.0 -a 0 -b 0.8 --nce_t 0.07 \
--trial 1


# CKTF
CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19_2relus' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 --nce_t 0.07 \
--trial 1