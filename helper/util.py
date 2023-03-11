from __future__ import print_function

import torch
import numpy as np
import os
import json
import sys

def presnet_paramters(model):
    for name, p in model.named_parameters():
        print(f"{name:50} | {str(p.shape):50} | {p.requires_grad}")
        
def log(opt, t):
    z = vars(opt).copy()
    z.update(t)
    logname = os.path.join(opt.save_folder, 'log.txt')
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(z) + '\n')
    print(z)

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer, logger):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr    
    
    # log lr
    for i, param_group in enumerate(optimizer.param_groups):
        logger.log_value(f'{i}_learning_rate_epoch', param_group['lr'], epoch)


def adjust_learning_rate_gradual(idx, curr_epoch, train_loader, opt, optimizer, logger):
    """Sets the learning rate to the initial LR decayed gradually in each step"""
    train_size = len(train_loader.dataset)
    steps_per_epoch = len(train_loader) # or: round(train_size / opt.batch_size)
    total_steps = opt.epochs * steps_per_epoch
    curr_step = (curr_epoch-1) * steps_per_epoch + idx
    
    stop_decay_epoch = 210
    if curr_epoch <= stop_decay_epoch:
        new_lr = 0.5 * opt.learning_rate * (1 + np.cos(np.pi * curr_step / total_steps))
    else:
        stop_decay_step = (stop_decay_epoch-1) * steps_per_epoch + len(train_loader)
        new_lr = 0.5 * opt.learning_rate * (1 + np.cos(np.pi * stop_decay_step / total_steps))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # log lr
    for i, param_group in enumerate(optimizer.param_groups):
        logger.log_value(f'{i}_learning_rate_step', param_group['lr'], curr_step)
        logger.log_value(f'{i}_learning_rate_epoch', param_group['lr'], curr_epoch)


def adjust_learning_rate_exp_with_warmup(idx, curr_epoch, train_loader, opt, optimizer, logger):
    """
    learning rate decay
    Args:
        initial_lr: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
        lr_steps: list of steps to apply exp_gamma
    """
    initial_lr = opt.learning_rate
    warmup_epochs = opt.warmup_epochs
    hold_epochs = opt.hold_epochs
    min_lr =  opt.min_lr
    exp_gamma = opt.lr_exp_gamma

    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    hold_steps = hold_epochs * steps_per_epoch
    curr_step = (curr_epoch-1) * steps_per_epoch + idx

    assert exp_gamma is not None

    if curr_step < warmup_steps:
        a = (curr_step + 1) / (warmup_steps + 1)
    elif curr_step < warmup_steps + hold_steps:
        a = 1.0
    else:
        a = exp_gamma ** (curr_epoch - warmup_epochs - hold_epochs)

    new_lr = max(a * initial_lr, min_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # log lr
    for i, param_group in enumerate(optimizer.param_groups):
        logger.log_value(f'{i}_learning_rate_step', param_group['lr'], curr_step)
        logger.log_value(f'{i}_learning_rate_epoch', param_group['lr'], curr_epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    pass
