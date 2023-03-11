
# ######################################################
# Transfer learning tiny-imagenet --> stl10
# VGG-19 --> VGG-19
# Step1: Train Student-fc200, using "unlabeled stl10"
# Step2: Fine-tune Fc of Student-fc10, using "train stl10"
########################################################



# ==================================Step1 and Step 2 together =========================
# CRD
CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "stl10" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg19 \
--model_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill crd -r 0.0 -a 0.0 -b 1.0 --nce_t 0.07 \
--trial 1 \
&& \
CUDA_VISIBLE_DEVICES=2 python3 train_teacher.py \
--dataset "stl10" \
--model_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19' \
--model vgg19 \
--fine_tune True \
--load_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19/S:vgg19_T:vgg19_stl10_crd_lrDecay_head:linear_r:0.0_a:0.0_b:1.0_theta:0.1_lr:0.05_lrDecayRate:0.1_lrDecayEpochs:[150, 180, 210]_init:False_t:0.07_1/vgg19_best.pth' \
--trial 1


# KD
CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--dataset "stl10" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg19 \
--model_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill kd -r 0.0 -a 1.0 -b 0.0 \
--trial 1 \
&& \
CUDA_VISIBLE_DEVICES=3 python3 train_teacher.py \
--dataset "stl10" \
--model_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19' \
--model vgg19 \
--fine_tune True \
--load_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19/S:vgg19_T:vgg19_stl10_kd_lrDecay_head:linear_r:0.0_a:1.0_b:0.0_theta:0.1_lr:0.05_lrDecayRate:0.1_lrDecayEpochs:[150, 180, 210]_init:False_t:0.1_1/vgg19_best.pth' \
--trial 1


# CKTF
CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "stl10" \
--path_t ./save/vanilla_model/tiny-imagenet/vgg19_vanilla/vgg19_best.pth --model_s vgg19 \
--model_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19' \
--batch_size 64 --learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--distill crdst --st_method Last -r 0.0 -a 0.0 -b 1.0 --theta 0.04 --nce_t 0.07 \
--trial 1 \
&& \
CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "stl10" \
--model_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19' \
--model vgg19 \
--fine_tune True \
--load_path './save/student_model/icassp2023/transfer_tiny-imagenet_to_stl10/S:vgg19_T:vgg19/S:vgg19_T:vgg19_stl10_crdstLast_lrDecay_head:linear_r:0.0_a:0.0_b:1.0_theta:0.04_lr:0.05_lrDecayRate:0.1_lrDecayEpochs:[150, 180, 210]_init:False_t:0.07_1/vgg19_best.pth' \
--trial 1


