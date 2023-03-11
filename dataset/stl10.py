from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import sys

# from torchvision import datasets, transforms # org
from torchvision import transforms
from torchvision_custom import datasets


"""
===================== Overview of STL10 ==================================
Refer to: https://ai.stanford.edu/~acoates/stl10/
10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
Images are 96x96 pixels, color.
500 training images (10 pre-defined folds), 800 test images per class.
100000 unlabeled images for unsupervised learning. These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set.
Images were acquired from labeled examples on ImageNet.

refer to: https://spell.ml/blog/simple-contrastive-learning-representation-using-the-X7QycRIAACQAqLu-
"""


def get_data_folder():
    data_folder = './data/'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class STL10Instance(datasets.STL10):
    """
    STL10Instance Dataset.
    Copy from __getitem__ in datasets.STL10
    """
    def __getitem__(self, index):        
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_stl10_dataloaders(traing_data_type="unlabeled", batch_size=128, num_workers=8, is_instance=False):
    """
    STL 10
    """
    
    data_folder = get_data_folder()
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    if is_instance:
        train_set = STL10Instance(root=data_folder,
                                     split=traing_data_type,
                                     download=True,
                                     transform=train_transform)
    else:
        train_set = datasets.STL10(root=data_folder,
                                    split=traing_data_type,
                                    download=True,
                                    transform=train_transform)
    if traing_data_type == "unlabeled":
        train_set.labels = np.asarray([0] * train_set.data.shape[0])
        print("traing_data_type is unlabeled, convert labels from -1 to 0")
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=data_folder,
                              split="test",
                              download=True,
                              transform=test_transform)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    print(f"\033[91mInside get_stl10_dataloaders, number of training data (100000 or 5000): {n_data}, number of testing data (8000): {len(test_set)}\033[00m")
    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class STL10InstanceSample(datasets.STL10):
    """
    STL10Instance+Sample Dataset
    10 classes
    Unlabeled data: 100,000
    Training data: 5000, 500 images per class
    Testing data: 8000,  800 images per class
    """
    def __init__(self, root, split='train', folds=None, transform=None, target_transform=None, download=False,
                 k=4096, mode='exact', is_sample=True, percent=1.0,):
        super().__init__(root, split=split, folds=folds, transform=transform, target_transform=target_transform, download=download)


        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10

        num_samples = len(self.data)
        label = self.labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive) # shape: (100, 500)
        self.cls_negative = np.asarray(self.cls_negative) # shape: (100, 49500)
        # print(self.cls_positive.shape, self.cls_negative.shape) 


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            # print(f"mode: {self.mode}")
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False # False; k:16384; len(self.cls_negative[target]): 49500
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace) # shape: (16384,)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx)) # shape: (16385,)

            return img, target, index, sample_idx


def get_stl10_dataloaders_sample(traing_data_type="unlabeled", batch_size=128, num_workers=8, k=4096, mode='exact', is_sample=True, percent=1.0):
    """
    stl10
    """

    data_folder = get_data_folder()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = STL10InstanceSample(root=data_folder,
                                    split=traing_data_type,
                                    download=True,
                                    transform=train_transform,
                                    k=k,
                                    mode=mode,
                                    is_sample=is_sample,
                                    percent=percent)
    if traing_data_type == "unlabeled":
        train_set.labels = np.asarray([0] * train_set.data.shape[0])
        print("traing_data_type is unlabeled, convert labels from -1 to 0")
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=data_folder,
                              split="test",
                              download=True,
                              transform=test_transform)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    return train_loader, test_loader, n_data
