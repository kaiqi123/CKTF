from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from xmlrpc.client import boolean

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from helper.util import log

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.stl10 import get_stl10_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet5', 'resnet5x4', 'resnet17x4', # new add
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 
                                 'ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tiny-imagenet', 'imagenet', 'stl10'], help='dataset')
    parser.add_argument('--model_path', type=str, default='./save/vanilla_models', help='the path where the model is saved')
    parser.add_argument('--distill', type=str, default='vanilla', choices=['vanilla'], help='to distinguish from mr')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # new add for fine-tune
    parser.add_argument('--fine_tune', type=boolean, default=False, help='set to true if fine tune')
    parser.add_argument('--load_path', type=str, help='the path where the pre-trained model is saved')


    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2'] and opt.dataset == "cifar10":
        opt.learning_rate = 0.1

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f'{opt.dataset}_{opt.model}_lr:{opt.learning_rate}_lrDecayEpochs:{opt.lr_decay_epochs}_'\
                     f'epochs:{opt.epochs}_weightDecay:{opt.weight_decay}_trial:{opt.trial}'
    if opt.fine_tune:
        kd_method_list = opt.load_path.split("/")[-2].split("_")
        if "vanilla" in kd_method_list:
            kd_method_name = kd_method_list[-1]  
        else:
            kd_method_name = kd_method_list[3] + "_" + kd_method_list[8] + "_" + kd_method_list[9]
        opt.model_name = kd_method_name + "_" + opt.model_name
    print(opt.learning_rate, opt.batch_size, opt.lr_decay_epochs, opt.lr_decay_rate, opt.epochs)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.tb_folder = os.path.join(opt.save_folder, opt.model_name+"_tensorboards")
    printRed(f"===>Save model to {opt.save_folder}\n===>Save tensorboards to: {opt.tb_folder}")

    return opt


def printRed(skk): print("\033[91m{}\033[00m" .format(skk))


def main():
    total_start_time = time.time()
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_target_labels = [e for e in range(0, 100)]
        test_target_labels = [e for e in range(0, 100)]
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, \
            is_instance=False, train_target_labels=train_target_labels, test_target_labels=test_target_labels)                                                       
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    elif opt.dataset == 'imagenet' or opt.dataset == "tiny-imagenet":
        train_loader, val_loader, n_cls = get_imagenet_dataloader(opt.dataset, opt.batch_size, opt.num_workers) 
    elif opt.dataset == 'stl10':
        train_loader, val_loader = get_stl10_dataloaders("train", opt.batch_size, opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)
    printRed(f"Dataset: {opt.dataset}, number of classes: {n_cls}")

    # model
    model = model_dict[opt.model](num_classes=n_cls)
    num_parameters = sum(p.numel() for p in model.parameters())
    printRed(f"Total num_parameters: {num_parameters}")

    if opt.fine_tune:
        # load parameters
        printRed(f"Loaded parameters from: {opt.load_path}")
        trained_model = torch.load(opt.load_path)
        current_dict = model.state_dict()
        count_copied_parameters = 0
        for key in trained_model['model'].keys():
            assert key in current_dict.keys()
            if key != "linear.weight" and key != "linear.bias" \
                and key != "classifier.weight" and key != "classifier.bias" \
                    and key != "fc.weight" and key != "fc.bias":
                current_dict[key].copy_(trained_model['model'][key])
                count_copied_parameters += 1
                # print(key)
        model.load_state_dict(current_dict)
        loaded_legnth = len(trained_model['model'].keys())
        printRed(f"Loaded number of modules: {loaded_legnth}, Copied: {count_copied_parameters}")

        # freeze parameters
        for name, param in model.named_parameters():
            if name != "linear.weight" and name != "linear.bias" \
                and name != "classifier.weight" and name != "classifier.bias" \
                    and name != "fc.weight" and name != "fc.bias":
                param.requires_grad = False
            # print(name, param.requires_grad)
        printRed("Following is trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        printRed(f"Total num_trainable_parameters: {num_trainable_parameters}")

    # optimizer
    nesterov_flag = True if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2'] else False
    optimizer = optim.SGD(model.parameters(),
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay,
                        nesterov=nesterov_flag) 

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer, logger)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt, logger)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        # if epoch % opt.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model.state_dict(),
        #         'accuracy': test_acc,
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)

        # save log.txt in each epoch
        eslaped_time = round((time.time() - total_start_time)/3600.0, 2)
        train_time = round(time2 - time1, 2)
        save_log_dict = {
            "curr_epoch": epoch,
            "test_acc_top1": round(test_acc.item(), 2),
            "test_acc_top5": round(test_acc_top5.item(), 2),
            "best_acc": round(best_acc.item(), 2),
            "train_acc": round(train_acc.item(), 2),
            "test_loss": round(test_loss, 2),
            "train_loss": round(train_loss, 2),
            "n_parameters": num_parameters,
            "train_time": train_time,
            "eslaped_time": eslaped_time,
        }
        log(opt, save_log_dict)
        print('==> model_name: %s (%d/%d), test_acc_top1: \33[91m%.2f\033[0m, best_acc: \33[91m%.2f\033[0m' %
                (opt.model_name, epoch, opt.epochs, test_acc, best_acc))

    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': test_acc,
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
