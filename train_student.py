"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
from pickle import NONE
import socket
import time
import sys

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_imagenet_dataloader_sample
from dataset.stl10 import get_stl10_dataloaders, get_stl10_dataloaders_sample

from helper.util import adjust_learning_rate
from helper.util import log

from distiller_zoo import SimilarityTransfer
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    hostname = socket.gethostname()
    print(f"hostname: {hostname}")

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=240, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tiny-imagenet', 'stl10'], help='dataset')
    parser.add_argument('--num_train_categories', type=int, default=100, help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet5', 'resnet5x4', 'resnet17x4', # new add
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
                                 'ResNet10', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--model_path', type=str, default='./save/student_model', help='the path where the student model is saved')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'rmseSt', 'rmseStcrd', 'crdst','hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses or crd loss')
    parser.add_argument('-theta', '--theta', type=float, default=0.1, help='weight balance for crdSt losses')
    parser.add_argument('-theta_epoch', '--theta_epoch', type=int, default=300, help='the epoch when theta change to 0.0')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.1, type=float, help='temperature parameter for softmax') # org: 0.07, 0.1 for cifar100, 0.07 for imagenet
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--head', default='linear', type=str, choices=['linear', 'mlp', 'pad']) # new add

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    
    # ST
    parser.add_argument('--st_method', type=str, default='Smallest', choices=['Last', 'Smallest', 'Largest', 'First', 'Random'])
    
    # second kd distillation, default is crdst
    parser.add_argument('--distill2', type=str, default='', choices=['crdst', '']) # new add
    parser.add_argument('--kd2weights', type=float, default=0.8, help='weight balance for second losse, default is crdst') # new add
    
    # whether to initialize
    parser.add_argument('--init_flag', default=False, type=bool, choices=[True, False])
 

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)
    printRed(f"Teacher name: {opt.model_t}")
    
    save_method_name = opt.distill + opt.st_method if 'st' in opt.distill or 'St' in opt.distill else opt.distill
    save_method_name = save_method_name + "_lrDecay"   

    if opt.distill2 == '':
        # 1 kd method
        opt.model_name = f'S:{opt.model_s}_T:{opt.model_t}_{opt.dataset}_{save_method_name}'\
            f'_head:{opt.head}_featDim:{opt.feat_dim}_mode:{opt.mode}_r:{opt.gamma}_a:{opt.alpha}_b:{opt.beta}_theta:{opt.theta}'\
            f'_lr:{opt.learning_rate}_lrDecayRate:{opt.lr_decay_rate}_lrDecayEpochs:{opt.lr_decay_epochs}_init:{opt.init_flag}'\
            f'_t:{opt.nce_t}_{opt.trial}'
    else:
        # 2 kd methods
        opt.model_name = f'S:{opt.model_s}_T:{opt.model_t}_{opt.dataset}_{save_method_name}'\
            f'_KD2:{opt.distill2}{opt.st_method}_kd2weights:{opt.kd2weights}'\
            f'_r:{opt.gamma}_a:{opt.alpha}_b:{opt.beta}'\
            f'_lr:{opt.learning_rate}_lrDecayRate:{opt.lr_decay_rate}_lrDecayEpochs:{opt.lr_decay_epochs}_init:{opt.init_flag}'\
            f'_t:{opt.nce_t}_{opt.trial}'
    
    opt.model_name = "test_"+opt.model_name
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.tb_folder = os.path.join(opt.save_folder, opt.model_name+"_tensorboards")
    printRed(f"===>Save model to {opt.save_folder}\n===>Save tensorboards to: {opt.tb_folder}")
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def printRed(skk): print("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m{}\033[00m" .format(skk))

def main():

    best_acc = 0
    total_start_time = time.time()

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':

        train_target_labels = [e for e in range(0, opt.num_train_categories)]
        test_target_labels = [e for e in range(0, 100)]

        if 'crd' in opt.distill or 'crd' in opt.distill2:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode,
                                                                               train_target_labels=train_target_labels,
                                                                               test_target_labels=test_target_labels)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,
                                                                        train_target_labels=train_target_labels,
                                                                        test_target_labels=test_target_labels)
        n_cls = 100
    elif opt.dataset == 'imagenet' or opt.dataset == "tiny-imagenet":
        if 'crd' in opt.distill or 'crd' in opt.distill2:
            train_loader, val_loader, n_data, n_cls = get_imagenet_dataloader_sample(
                dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, is_sample=True, k=opt.nce_k)
        else:
            train_loader, val_loader, n_data, n_cls = get_imagenet_dataloader(
                dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True) 
    
    elif opt.dataset == 'stl10':
        if 'crd' in opt.distill or 'crd' in opt.distill2:
            train_loader, val_loader, n_data = get_stl10_dataloaders_sample(
                traing_data_type="unlabeled",
                batch_size=opt.batch_size, num_workers=opt.num_workers, k=opt.nce_k, mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_stl10_dataloaders(
                traing_data_type="unlabeled",
                batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
        n_cls = 200  
    printRed(f"Dataset: {opt.dataset}, number of training data: {n_data}, number of classes: {n_cls}")
    
    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    
    # for one-step transfer learning, teacher-fc200 -> student-fc10 (Tiny-ImageNet --> STL-10)
    # model_t = load_teacher(opt.path_t, 200)
    # model_s = model_dict[opt.model_s](num_classes=10)
    
    # calculate and show paramters
    # def presnet_paramters(model):
    #     for name, p in model.named_parameters():
    #         print(f"{name:50} | {str(p.shape):50} | {p.requires_grad}")
    # print("\nStudent ==>"); presnet_paramters(model_s)
    # print("\nTeacher ==>"); presnet_paramters(model_t)
    num_parameters_s = sum(p.numel() for p in model_s.parameters())
    num_parameters_t = sum(p.numel() for p in model_t.parameters())
    print(f'Total number of parameters ==>\t student: {num_parameters_s}, teacher: {num_parameters_t}, Compression Ratio: {num_parameters_t/num_parameters_s:.2f}')


    if opt.distill2 == "":
        printRed("===> Single KD")
        flatGroupOut = True if opt.distill == 'crdst' else False
        data = torch.randn(2, 3, 32, 32)
        model_t.eval()
        model_s.eval()
        feat_t, block_out_t, _ = model_t(data, is_feat=True, flatGroupOut=flatGroupOut, kd2=False)
        feat_s, block_out_s, _ = model_s(data, is_feat=True, flatGroupOut=flatGroupOut, kd2=False)
    else:
        printRed("===> Two KDs")
        data = torch.randn(2, 3, 32, 32)
        model_t.eval()
        model_s.eval()
        feat_t, feat_t2, block_out_t, _ = model_t(data, is_feat=True, flatGroupOut=False, kd2=True)
        feat_s, feat_s2, block_out_s, _ = model_s(data, is_feat=True, flatGroupOut=False, kd2=True)


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data # number of training data, cifar100: 50000
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'crdst':
        similarity_transfer = SimilarityTransfer(opt.st_method, opt.model_s)
        criterion_kd = nn.ModuleList([])
        criterion_kd.append(similarity_transfer)
        for i in range(len(feat_s)): 
            if i < len(feat_s)-1: 
                opt.s_dim = feat_t[i].shape[1]
            else: 
                opt.s_dim = feat_s[i].shape[1]
            opt.t_dim = feat_t[i].shape[1]
            opt.n_data = n_data
            criterion_kd_single = CRDLoss(opt)
            module_list.append(criterion_kd_single.embed_s)
            module_list.append(criterion_kd_single.embed_t)
            trainable_list.append(criterion_kd_single.embed_s)
            trainable_list.append(criterion_kd_single.embed_t)
            criterion_kd.append(criterion_kd_single)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)


    if opt.distill2 == 'crdst':
        printRed("===> opt.distill2 is crdst")
        criterion_kd2 = nn.ModuleList([])
        criterion_kd2.append(SimilarityTransfer(opt.st_method, opt.model_s))
        for i in range(len(feat_s2)): 
            if i < len(feat_s2)-1: 
                opt.s_dim = feat_t2[i].shape[1]
            else:
                opt.s_dim = feat_s2[i].shape[1]
            opt.t_dim = feat_t2[i].shape[1]
            opt.n_data = n_data
            criterion_kd_single2 = CRDLoss(opt)
            module_list.append(criterion_kd_single2.embed_s)
            module_list.append(criterion_kd_single2.embed_t)
            trainable_list.append(criterion_kd_single2.embed_s)
            trainable_list.append(criterion_kd_single2.embed_t)
            criterion_kd2.append(criterion_kd_single2)
            print(n_data)
            print(i, feat_s[i].shape, opt.s_dim, criterion_kd_single2.embed_s.linear)
            print(i, feat_t[i].shape, opt.t_dim, criterion_kd_single2.embed_t.linear)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    if opt.distill2 == 'crdst':
        criterion_list.append(criterion_kd2)     # the second knowledge distillation loss


    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    printRed(f"Initial learning rate: {opt.learning_rate}")

    # append teacher after optimizer to avoid weight_decay 
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True


    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    printRed(f'teacher accuracy: {teacher_acc}')


    # initialize student weights
    if opt.init_flag:
        printRed("===>Initialize student weights by copying teacher's weights")
        student_acc, _, _ = validate(val_loader, module_list[0], criterion_cls, opt)
        print('Before initialization, student_acc accuracy: ', student_acc)
        
        for s_name, s_para in module_list[0].named_parameters():
            for t_name, t_para in model_t.named_parameters():
                if s_name == t_name and "conv" in s_name:
                    assert s_para.shape == t_para.shape
                    s_para.data = t_para.data
                    print(s_name, s_para.shape, s_para.requires_grad)
        
        student_acc, _, _ = validate(val_loader, module_list[0], criterion_cls, opt)
        print('After initialization, student_acc accuracy: ', student_acc)
    else:
        printRed("===>Not initialize student weights by copying teacher's weights")

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer, logger)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, logger)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc >= best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        # if epoch % opt.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'accuracy': test_acc,
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)
        
        # save log.txt in each epoch
        eslaped_time = round((time.time() - total_start_time)/3600.0, 2)
        train_time = round(time2 - time1, 2)
        save_log_dict = {
            "curr_epoch": epoch,
            "test_acc_top1": round(test_acc.item(), 2),
            "test_acc_top5": round(tect_acc_top5.item(), 2),
            "best_acc": round(best_acc.item(), 2),
            "train_acc": round(train_acc.item(), 2),
            "test_loss": round(test_loss, 2),
            "train_loss": round(train_loss, 2),
            "teacher_acc_top1": round(teacher_acc.item(), 2),
            "n_parameters_student": num_parameters_s,
            "n_parameters_teacher": num_parameters_t,
            "train_time": train_time,
            "eslaped_time": eslaped_time,
        }
        log(opt, save_log_dict)
        print('==> model_name: %s (%d/%d), test_acc_top1: \33[91m%.2f\033[0m, best_acc: \33[91m%.2f\033[0m' %
                (opt.model_name, epoch, opt.epochs, test_acc, best_acc))

    print('best accuracy:', best_acc)

    # save last model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'accuracy': test_acc,
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
