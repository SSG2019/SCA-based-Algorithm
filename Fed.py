#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from joint import optimize
from models.BPDecoding import per_pkt_transmission_Proposed, per_pkt_transmission_Ori_ML, per_pkt_transmission_Ori_AS, \
    per_pkt_transmission_OFDMA, per_pkt_transmission_OAC, per_pkt_transmission_RIS, per_pkt_transmission_LMMSE
from utils.Ignore import ToIgnore, flatten, plot_pdf


# ================================================ perfect channel ->->-> no misalignments, no noise

def FedAvg(w, args, flag=0):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        # get the receivd signal
        if (flag == 1) and (k not in ToIgnore):
            continue
        # 补上之前在train中除去的学习率
        w_avg[k] = w_avg[k] * args.lr
        for i in range(1, len(w)):  # 叠上剩下3个用户的
            w_avg[k] += w[i][k] * args.lr

        # weight_coLector.append((w_avg[k]-noise).cpu().numpy())

        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# ================================================ Asynchronous AirComp
def FedAvg_ML(w, args, method, Q, P, V, idxs_users):
    L = args.L
    # w 是一个列表，每个元素是active用户的梯度： OrderedDict
    M = len(w)  # number of devices，M = 4

    # -------------- Extract the symbols (weight updates) from each devices as a numpy array (complex sequence)
    StoreRecover = np.array([])  # record how to transform the numpy arracy back to dictionary
    for m in np.arange(M):  # 对每一个active用户，
        # extract symbols from one device (layer by layer)
        wEach = w[m]  # 得到该用户的所有模参 OrderedDict
        eachWeightNumpy = np.array([])
        for k in wEach.keys():  # 对该用户某一层网络
            # The batch normalization layers should be ignroed as they are not weights
            # (can be transmitted reliably to the PS in practice)
            if k in ToIgnore:
                continue  # 如果这一个键在”ToIgnore“中（是batch normalization layers），就跳过此次循环
            temp = wEach[k].cpu().numpy()  # 得到这一层的具体权重值
            temp, unflatten = flatten(temp)  # 将本层的模参转为 ndarray 并展开
            if m == 0:  # 只记录某一个用户，如用户1的，每层模参展开之前的形状，以便后续恢复
                StoreRecover = np.append(StoreRecover, unflatten)  # StoreRecover是一个类型为 object 的 ndarray，每一个元素都是一个函数
            eachWeightNumpy = np.append(eachWeightNumpy, temp)  # 该用户的模参向量：ndarray(21840,)！！！

        # stack the symbols from different devices ->-> numpy array of shape M * d_symbol = 4 * 10920
        complexSymbols = eachWeightNumpy[0:int(len(eachWeightNumpy) / 2)] + 1j * \
                         eachWeightNumpy[int(len(eachWeightNumpy) / 2):]
        # 前2/d的模参为实部，后2/d的模参为虚部，组成一个长度为2/d的复数类型的 ndarray(10920,)
        if m == 0:
            TransmittedSymbols = np.array([complexSymbols])
        else:
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([complexSymbols])]  # ndarray(4, 10920)，active用户的模参
            # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
            # m == 0时，说明是第一个用户的模参，直接创建；m != 0 则是后面的用户，需要用np.r_拼接
            # TransmittedSymbols：ndarray(4, 10920)，active用户的模参

    # number of complex symbols from each device（第1行的维度）
    d_symbol = len(TransmittedSymbols[0])  # 需要传输10920个符号，一个 time slot 传输 L 个 symbols

    # ---------------------------------------------------------------------------------- pkt by pkt transmission
    # add 1 all-zero column => 631927 + 1 = 631928 (11 * 4 * 43 * 167 * 2) = 57448 * 11 or 28724 * 22
    # TransmittedSymbols = np.c_[TransmittedSymbols, np.zeros([M, 1])]  # 4 * 10921（为了满足包长，水平加了一个全零列）

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        numPkt = 42  # 数据包个数: 42
        lenPkt = int(d_symbol / numPkt)  # 包长度: 260

    elif args.dataset == 'cifar':
        TransmittedSymbols = np.c_[TransmittedSymbols, np.zeros([M, 7])]
        numPkt = 424  # 数据包个数: 424
        lenPkt = int((d_symbol + 7) / numPkt)  # 包长度: 260

    if method == 'Proposed_SCA':  # proposed
        results = []
        for idx in range(numPkt):  # 对于所有用户的第idx个数据包来说
            # transmissted complex symbols in one pkt
            # 把所有用户的symbol划分为packet，按包发送，共260个包（需要 260 个 time slot）
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)

            # received complex symbols in each pkt (after estmimation and averaging)
            symbols_received = per_pkt_transmission_Proposed(args, L, copy.deepcopy(onePkt), Q, P, V, idxs_users, idx)
            # symbols_received = per_pkt_transmission_Proposed(args, L, copy.deepcopy(onePkt), Q, P, V, idxs_users, dd, h)
            results.append(symbols_received)  # results就是最终接收到的数据包，是一个列表42个元素，每个元素是 ndarray:(260, )

        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)  # 将数据包恢复成原symbol，个数为：(10920, )

        # ReceivedComplexPkt = ReceivedComplexPkt + g_bar
        # ReceivedComplexPkt = ReceivedComplexPkt * args.lr
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        # Restore the real weights ->->-> numpy array
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))  # 恢复实部虚部，变为原本21840
        # ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:-1]), np.imag(ReceivedComplexPkt[:-1]))

        # the last element (0) must be deleted

        # Reconstruct the dictionary from the numpy array
        # run the federated averaging first (to tackle the batched normalization layer)
        # 对每个用户的模型参数变化量 / 梯度，用FedAvg算法，得到有序字典形式的-所有用户平均的模型参数变化量 / 梯度
        # 注意：这里是真实（原始）的梯度 w ！（其实就是借用一下w的结构，把 ReceivedPkt 中对应的值复制过去。。）
        w_avg = FedAvg(w, args, 1)

        startIndex = 0
        # idx = 0
        for idx, k in enumerate(w_avg.keys()):  # 遍历模参的每一层
            # only update the non-batched-normalization-layers in w_avg
            if k not in ToIgnore:  # 如果该层不是归一化层，w_avg[k]第k层的权重，如：Tensor(10, 1, 5, 5)
                lenLayer = w_avg[k].numel()  # 本层的权重数量
                # get data
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]  # 找到对应的接收到的梯度
                # reshape
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)  # 利用之前保存的 StoreRecover 恢复形状
                # convert to torch in cuda()
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()  # 复制到 w （联邦平均后）的相应位置
                startIndex += lenLayer
                # idx += 1

    elif method == 'Ori_ML':  # Ori_ML
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_Ori_ML(args, L, copy.deepcopy(onePkt), V, idxs_users)
            # symbols_received = per_pkt_transmission_Ori_ML(args, L, copy.deepcopy(onePkt), V, idxs_users, dd, h)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):  # 遍历模参的每一层
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    elif method == 'Ori_AS':  # aligned_sample estimator
        results = []
        for idx in range(numPkt):  # 对于所有用户的第idx个数据包来说
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_Ori_AS(args, L, copy.deepcopy(onePkt), idxs_users)
            # symbols_received = per_pkt_transmission_Ori_AS(args, L, copy.deepcopy(onePkt), idxs_users, dd, h)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    elif method == 'OFDMA':
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_OFDMA(args, L, copy.deepcopy(onePkt), V, idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):  # 遍历模参的每一层
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    elif method == 'OAC':
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_OAC(args, L, copy.deepcopy(onePkt), V, idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):  # 遍历模参的每一层
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    elif method == 'Proposed_RIS':  # proposed
        results = []
        for idx in range(numPkt):  # 对于所有用户的第idx个数据包来说

            # transmissted complex symbols in one pkt
            # 把所有用户的symbol划分为packet，按包发送，共260个包（需要 260 个 time slot）
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)

            # received complex symbols in each pkt (after estmimation and averaging)
            symbols_received = per_pkt_transmission_RIS(args, L, copy.deepcopy(onePkt), Q, P, V, idxs_users,
                                                        idx)
            # symbols_received = per_pkt_transmission_Proposed(args, L, copy.deepcopy(onePkt), Q, P, V, idxs_users, dd, h)
            results.append(symbols_received)  # results就是最终接收到的数据包，是一个列表42个元素，每个元素是 ndarray:(260, )

        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)  # 将数据包恢复成原symbol，个数为：(10920, )

        # ReceivedComplexPkt = ReceivedComplexPkt + g_bar
        # ReceivedComplexPkt = ReceivedComplexPkt * args.lr
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        # Restore the real weights ->->-> numpy array
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]),
                                np.imag(ReceivedComplexPkt[:]))  # 恢复实部虚部，变为原本21840
        # ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:-1]), np.imag(ReceivedComplexPkt[:-1]))

        # the last element (0) must be deleted

        # Reconstruct the dictionary from the numpy array
        # run the federated averaging first (to tackle the batched normalization layer)
        # 对每个用户的模型参数变化量 / 梯度，用FedAvg算法，得到有序字典形式的-所有用户平均的模型参数变化量 / 梯度
        # 注意：这里是真实（原始）的梯度 w ！（其实就是借用一下w的结构，把 ReceivedPkt 中对应的值复制过去。。）
        w_avg = FedAvg(w, args, 1)

        startIndex = 0
        # idx = 0
        for idx, k in enumerate(w_avg.keys()):  # 遍历模参的每一层
            # only update the non-batched-normalization-layers in w_avg
            if k not in ToIgnore:  # 如果该层不是归一化层，w_avg[k]第k层的权重，如：Tensor(10, 1, 5, 5)
                lenLayer = w_avg[k].numel()  # 本层的权重数量
                # get data
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]  # 找到对应的接收到的梯度
                # reshape
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)  # 利用之前保存的 StoreRecover 恢复形状
                # convert to torch in cuda()
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()  # 复制到 w （联邦平均后）的相应位置
                startIndex += lenLayer
                # idx += 1

    elif method == 'LMMSE':  # LMMSE
        results = []
        for idx in range(numPkt):
            onePkt = TransmittedSymbols[:, (idx * lenPkt):((idx + 1) * lenPkt)]  # (4, 260)
            symbols_received = per_pkt_transmission_LMMSE(args, L, copy.deepcopy(onePkt), V, idxs_users)
            results.append(symbols_received)
        for idx in range(len(results)):
            output = results[idx]
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
        if args.dataset == 'cifar':
            ReceivedComplexPkt = ReceivedComplexPkt[:-7]
        ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:]), np.imag(ReceivedComplexPkt[:]))
        w_avg = FedAvg(w, args, 1)
        startIndex = 0
        for idx, k in enumerate(w_avg.keys()):  # 遍历模参的每一层
            if k not in ToIgnore:
                lenLayer = w_avg[k].numel()
                ParamsLayer = ReceivedPkt[startIndex:(startIndex + lenLayer)]
                ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
                w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()
                startIndex += lenLayer

    else:
        w_avg = None

    return w_avg
