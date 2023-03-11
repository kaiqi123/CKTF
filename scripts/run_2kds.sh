
# ======================================= 2 KDs: Ours (CKTF) + Others =================================
# resnet32x4 / resnet8x4
# cifar100
# batch_size: default 64
# =====================================================================================================

# FitNet + CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill hint -r 1.0 -a 0 -b 100 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1 

# AT + CKTF
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill attention -r 1.0 -a 0 -b 1000 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1 


# SP + CKTF
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill similarity -r 1.0 -a 0 -b 3000 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1 


# CC + CKTF
CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill correlation -r 1.0 -a 0 -b 0.02 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1 


# VID + CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill vid -r 1.0 -a 0 -b 1 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


# RKD + CKTF
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill rkd -r 1.0 -a 0 -b 1 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


# PKT + CKTF
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/rn32_rn8' \
--distill pkt -r 1.0 -a 0 -b 30000 \
--distill2 crdst --st_method Smallest --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


# AB + CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/icassp2023/cifar100/rn32X4_rn8X4/2kds_last' \
--distill abound -r 1.0 -a 0 -b 1 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


# FT + CKTF
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/icassp2023/cifar100/rn32X4_rn8X4/2kds_last' \
--distill factor -r 1.0 -a 0 -b 200 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


# NST + CKTF
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
--model_s resnet8x4 --model_path './save/student_model/icassp2023/cifar100/rn32X4_rn8X4/2kds_last' \
--distill nst -r 1.0 -a 0 -b 50 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1



# ======================================= 2 KDs: Ours (CKTF) + Others =================================
# vgg19/vgg8
# Tiny-ImageNet
# =====================================================================================================

# FitNet + CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill hint -r 1.0 -a 0 -b 100 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1

# AT + CKTF
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill attention -r 1.0 -a 0 -b 1000 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1

# SP + CKTF
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill similarity -r 1.0 -a 0 -b 3000 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


# CC + CKTF
CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill correlation -r 1.0 -a 0 -b 0.02 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1

# VID + CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill vid -r 1.0 -a 0 -b 1 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


# RKD + CKTF
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill rkd -r 1.0 -a 0 -b 1 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


# PKT + CKTF
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill pkt -r 1.0 -a 0 -b 30000 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


# AB + CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill abound -r 1.0 -a 0 -b 1 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


# FT + CKTF
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill factor -r 1.0 -a 0 -b 200 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


# NST + CKTF, bs is set to 32, due to out of memory
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "tiny-imagenet" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg8 \
--model_path './save/student_model/icassp2023/tiny-imagenet/S:vgg8_T:vgg19/2kds' \
--batch_size 32 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill nst -r 1.0 -a 0 -b 50 \
--distill2 crdst --st_method Last --kd2weights 0.8 \
--trial 1


echo "finish student"
exit
