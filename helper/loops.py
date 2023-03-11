from __future__ import print_function, division

import sys
import time
import torch
import copy 

from .util import AverageMeter, accuracy
from .util import adjust_learning_rate_gradual, adjust_learning_rate_exp_with_warmup

from .util import presnet_paramters



def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt, logger, replacing_rate_scheduler=None):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader): # org

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()

        if torch.isnan(loss).any():
            print(f'WARNING: loss is NaN; skipping update')
            sys.exit()

        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg





def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, logger):
    """One epoch distillation"""

    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        
        if 'crd' in opt.distill or 'crd' in opt.distill2:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)


        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if 'crd' in opt.distill or 'crd'in opt.distill2:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        if opt.distill2 == "":
            # for 1 kd
            flatGroupOut = True if opt.distill == 'crdst' else False # for crdst
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, block_out_s, logit_s = model_s(input, is_feat=True, preact=preact, flatGroupOut=flatGroupOut, kd2=False)
            with torch.no_grad():
                feat_t, block_out_t, logit_t = model_t(input, is_feat=True, preact=preact, flatGroupOut=flatGroupOut, kd2=False)
                feat_t = [f.detach() for f in feat_t]
        else:
            # for 2 kds
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, feat_s2, block_out_s, logit_s = model_s(input, is_feat=True, preact=preact, flatGroupOut=False, kd2=True)
            with torch.no_grad():
                feat_t, feat_t2, block_out_t, logit_t = model_t(input, is_feat=True, preact=preact, flatGroupOut=False, kd2=True)
                feat_t = [f.detach() for f in feat_t]
                feat_t2 = [f.detach() for f in feat_t2]


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'crdst':
            layer_pair_list = criterion_kd[0](block_out_s, block_out_t)

            loss_kd_crdSt_list = []

            # for f0
            f0_s = feat_s[0]
            f0_t = feat_t[0]
            loss_kd_crdSt_list.append(criterion_kd[1](f0_s, f0_t, index, contrast_idx))
            
            # for f1,f2,f3 or for f1,f2,f3,f4
            for i in range(2, len(layer_pair_list)+2): # for f1,f2,f3 or for f1,f2,f3,f4
            # for i in range(2, len(layer_pair_list)+1): # for f1,f2 or for f1,f2,f3
                f_s, f_t = layer_pair_list[i-2]
                loss_kd_crdSt_list.append(criterion_kd[i](f_s, f_t, index, contrast_idx))

            # for f4 or for f5
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd_crd = criterion_kd[-1](f_s, f_t, index, contrast_idx)
            
            # for using theta
            loss_kd_crdSt = sum(loss_kd_crdSt_list)

        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)


        # for crdst, using opt.theta and opt.theta_epoch to control whether to add crdst 
        if opt.distill == 'crdst':
            # opt.theta = opt.theta if epoch < opt.theta_epoch else 0.06
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd_crd + opt.theta * loss_kd_crdSt
            logger.log_value(f'theta', opt.theta, epoch)
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        if idx == 0:
            print("\033[91m{}\033[00m" .format(f"opt.gamma: {opt.gamma}, opt.alpha: {opt.alpha}, opt.beta: {opt.beta}, opt.theta: {opt.theta}"))

        if opt.distill2 == 'crdst':
            criterion_kd2 = criterion_list[3]
            
            layer_pair_list = criterion_kd2[0](block_out_s, block_out_t)

            loss_kd_crd_list = []
            loss_kd_crdSt_list = []

            # print("=====>for f0")
            f0_s2 = feat_s2[0]
            f0_t2 = feat_t2[0]
            loss_kd_crdSt_list.append(criterion_kd2[1](f0_s2, f0_t2, index, contrast_idx))
            
            # print("======>for f1,f2,f3")
            for i in [2,3,4]:
                f_s2, f_t2 = layer_pair_list[i-2]
                loss_kd_crdSt_list.append(criterion_kd2[i](f_s2, f_t2, index, contrast_idx))
                
            # print("=====>for f4")
            f_s2 = feat_s2[-1]
            f_t2 = feat_t2[-1]
            loss_kd_crd_list.append(criterion_kd2[-1](f_s2, f_t2, index, contrast_idx))
                
            loss_kd_crdSt = sum(loss_kd_crdSt_list)
            loss_kd_crd = sum(loss_kd_crd_list)
            loss_kd2 =  loss_kd_crdSt + loss_kd_crd          
            loss = loss + opt.kd2weights * loss_kd2


        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        
        # new add, check whether loss is NaN
        if torch.isnan(loss).any():
            print(f'WARNING: loss is NaN; skipping update')
            sys.exit()      

        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    

    return top1.avg, losses.avg



def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
