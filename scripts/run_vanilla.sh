# ==========================================================Tiny-ImageNet=================================================================

CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "tiny-imagenet" --model ResNet10 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "tiny-imagenet" --model ResNet34 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=1 python3 train_teacher.py \
--dataset "tiny-imagenet" --model ResNet50 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=2 python3 train_teacher.py \
--dataset "tiny-imagenet" --model ResNet101 --model_path './save/vanilla_model/tiny-imagenet' --batch_size 32 --trial 1 

CUDA_VISIBLE_DEVICES=3 python3 train_teacher.py \
--dataset "tiny-imagenet" --model ResNet152 --model_path './save/vanilla_model/tiny-imagenet' --batch_size 16 --trial 1 

CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "tiny-imagenet" --model vgg8 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=1 python3 train_teacher.py \
--dataset "tiny-imagenet" --model vgg11 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=2 python3 train_teacher.py \
--dataset "tiny-imagenet" --model vgg13 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=3 python3 train_teacher.py \
--dataset "tiny-imagenet" --model vgg16 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 

CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "tiny-imagenet" --model vgg19 --model_path './save/vanilla_model/tiny-imagenet' --trial 1 


# ======================================================= CIFAR-100 ====================================================================
CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "cifar100" --model vgg13 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=1 python3 train_teacher.py \
--dataset "cifar100" --model vgg8 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=1 python3 train_teacher.py \
--dataset "cifar100" --model resnet14 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=2 python3 train_teacher.py \
--dataset "cifar100" --model resnet8 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "cifar100" --model resnet32 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=1 python3 train_teacher.py \
--dataset "cifar100" --model resnet44 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py \
--dataset "cifar100" --model resnet5 --model_path './save/vanilla_model/cifar100' --trial 1 

CUDA_VISIBLE_DEVICES=1 python3 train_teacher.py \
--dataset "cifar100" --model resnet17x4 --model_path './save/vanilla_model/cifar100' --trial 1 


echo "finish vanilla"
exit

