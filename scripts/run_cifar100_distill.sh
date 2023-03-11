
# ======================================================== CKTF ===================================================================

CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/vanilla_model/cifar100/vgg13_vanilla/vgg13_best.pth \
--model_s vgg8 --model_path './save/student_model/icann2022/cifar100' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
--model_s resnet32 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
--model_s resnet20 --model_path './save/student_model/icann2022/cifar100' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
--model_s resnet20 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth \
--model_s wrn_40_1 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth \
--model_s wrn_16_2 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1

# ======================================================== CKTF + KD ===================================================================
CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/vanilla_model/cifar100/vgg13_vanilla/vgg13_best.pth \
--model_s vgg8 --model_path './save/student_model/icann2022/cifar100' \
--distill crdst --st_method Last -r 1.0 -a 1.0 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1

CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
--model_s resnet32 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 1 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=2 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
--model_s resnet20 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 1 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=3 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
--model_s resnet20 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 1 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=0 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth \
--model_s wrn_40_1 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 1 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1


CUDA_VISIBLE_DEVICES=1 python3 train_student.py \
--dataset "cifar100" \
--path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth \
--model_s wrn_16_2 --model_path './save/student_model/cifar100_new' \
--distill crdst --st_method Last -r 1.0 -a 1 -b 0.8 --theta 0.1 \
--learning_rate 5e-2 --lr_decay_rate 0.1 --lr_decay_epochs '150,180,210' \
--trial 1

