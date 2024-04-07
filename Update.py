#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random, pdb
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if self.args.local_bs == 0:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(idxs), shuffle=True)
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=1500, shuffle=True)

    def train(self, net, history_dict, user_idx, args):  # 模型，本次的全局模参，用户idx
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels).to(self.args.device)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 50 == 0:
                    print(f'User{user_idx}: Update Epoch: {iter}/{self.args.local_ep - 1} '
                          f'[{batch_idx * len(images)}/{len(self.ldr_train.dataset)} '
                          f'({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss.item():.6f}')
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        current_dict = net.state_dict()

        for k in current_dict.keys():
            current_dict[k] -= history_dict[k]  # 得到 模参的变化量 * lr
            current_dict[k] = current_dict[k] / args.lr  # 得到 模参的变化量
            # if proposed:
            #     current_dict[k] = current_dict[k] / args.lr   # 得到 模参的变化量

        return current_dict, sum(epoch_loss) / len(epoch_loss)
