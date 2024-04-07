#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import time
from load_data import Load_Data
import cvxpy as cp

from models.Fed import *
from models.Nets import CNNMnist, CNNCifar2
from models.Update import LocalUpdate
from models.test import test_img
from utils.options import args_parser
from utils.run_experiment_uitls import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Main(args, method):
    setup_seed(args.seed)
    M = max(int(args.frac * args.M_Prime), 1)  # M = 4
    L = args.L
    f = np.random.randn(5, 1) + np.random.randn(5, 1) * 1j
    f = f / np.linalg.norm(f, 2) * np.sqrt(1)
    f_norm2 = np.linalg.norm(f, 2) ** 2
    args.f_norm2 = f_norm2

    # taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    # taus[-1] = args.maxDelay
    # dd = np.zeros(M)
    # for idx in np.arange(M):
    #     if idx == 0:
    #         dd[idx] = taus[0]
    #     elif idx == M - 1:
    #         dd[idx] = args.T_sam - taus[-1]
    #     else:
    #         dd[idx] = taus[idx] - taus[idx - 1]
    # dd[dd < 1e-5] = 1e-5
    # h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)

    dict_users, dataset_train, dataset_test = Load_Data(args)  # 调试一下
    args.testDataset = dataset_test
    K = np.zeros(args.M_Prime)
    for i, item in enumerate(dict_users.keys()):
        K[i] = len(dict_users[item])
    args.K = K

    V = np.zeros((L, M * L))
    v = np.ones((1, M))
    # v[:] = v[:] * K[0] / (K[0] * 4)
    for i in range(L):
        V[i, np.arange(M) + i * M] = v
    P_entry = np.ones(M)
    P = np.zeros((M * (L + 1) - 1, M * L))
    for i in range(M * L):
        P[np.arange(M) + i, i] = P_entry[np.mod(i, M)]
    Q = V.T.conj() @ V

    result = {}

    # dict_users是一个字典，一个用户作为一个键值对（共40个），每一个值是一个集合，元素是该用户拥有的数据idx
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar2(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    w_glob = net_glob.state_dict()  # copy weights
    d = 0
    for item in w_glob.keys():
        d = d + int(np.prod(w_glob[item].shape))
    print(f"\nThe method = {method}")
    print("Running args.EsN0dB =", args.EsN0dB)
    print(f'Total Number of Parameters={d}')
    print(f'The Dataset = {args.dataset}')
    print(f'The Learning Rate = {args.lr}')
    print(f'The local E = {args.local_ep}')
    print(f'The Epoch = {args.epochs}')
    print(f'The iid = {args.iid}')
    print(f'Total RIS Parameters LL = {args.LL}')

    net_glob.train()
    print("============================== Federated Learning ... ...")
    # 创建两个模型，或者改回来，一次就跑一个代码，泡多个结果
    # training
    loss_train = []
    acc_store = []
    w_glob = net_glob.state_dict()  # initial global weights
    for iter in range(args.epochs):
        # record the running time of an iteration
        startTime = time.time()
        # setup_seed(1)
        # h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
        if iter == 0:
            net_glob.eval()
            acc_test, _ = test_img(net_glob, dataset_test, args)
            acc_store.append(acc_test.numpy())
            net_glob.train()
        history_dict = net_glob.state_dict()  # 保存本次迭代的初始模型

        # 随机选择 M * C个用户
        idxs_users = np.random.choice(range(args.M_Prime), M, replace=False)
        # ----------------------------------------------------------------------- Local Training
        w_locals = []  # store the local "updates" (the difference) of M devices
        loss_locals = []  # store the local training loss
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # 对于每一个用户创建一个local类
            # batch_size = 64；共15000个数据，则需要240个batch训练完成一个本地epoch
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), history_dict=history_dict,
                                  user_idx=idx, args=args)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))  # 对每一个active用户，将其更新的 w（模参变化量 / 梯度）添加到列表 w_locals 中
            loss_locals.append(copy.deepcopy(loss))  # 对每一个active用户，将其 loss 添加到列表 loss_locals 中
        # w_locals 是一个长度为 M 的列表
        # ----------------------------------------------------------------------- Federated Averaging
        if method == 'Noiseless':
            current_dict_var = FedAvg(w_locals, args)
            # 对模参变化量 w_locals 进行调整（w_locals是一个列表，每个元素是active用户的模参）
        else:
            current_dict_var = FedAvg_ML(w_locals, args, method, Q, P, V, idxs_users)
            # 对模参变化量 w_locals 进行调整（w_locals是一个列表，每个元素是active用户的模参）
        # current_dict_var 是平均后的：模参变化量/梯度
        # ----------------------------------------------------------------------- Reconstruct the new model at the PS
        for k in current_dict_var.keys():
            w_glob[k] = history_dict[k] + current_dict_var[k]
        # load new model
        net_glob.load_state_dict(w_glob)

        # print training loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(f'Round {iter:3d}, Average loss {loss_avg:.3f}, Time Cosumed {time.time() - startTime:.3f}')
        loss_train.append(loss_avg)

        # testing
        net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # acc_store = np.append(acc_store, acc_test.numpy())
        acc_store.append(acc_test.numpy())
        net_glob.train()
    result['train_loss'] = np.asarray(loss_train)
    result['test_acc'] = np.asarray(acc_store)

    # print('result {}'.format(result['Proposed_test_acc'][len(result['Proposed_test_acc']) - 1]))
    return result, net_glob


def runExp(name, args):
    if args.iid:
        isIID = 'iid'
    else:
        isIID = 'niid'
    if args.local_ep == 5:
        descFun = 'minibatch'
    else:
        descFun = 'fullbatch'

    method = 'Noiseless'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)

    # method = 'OFDMA'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_05_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'Ori_AS'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'Ori_ML'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'Proposed_SCA'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)


def runRISExp(name, args):
    if args.iid:
        isIID = 'iid'
    else:
        isIID = 'niid'
    if args.local_ep == 5:
        descFun = 'minibatch'
    else:
        descFun = 'fullbatch'

    # method = 'Noiseless'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'OFDMA'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_05_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'Ori_AS'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'Ori_ML'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)
    #
    # method = 'Proposed_SCA'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_{args.dataset}_{isIID}.npz'
    # np.savez(filename, vars(args), final_result)
    # modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    # torch.save(net_glob, modelFileName)

    method = 'Proposed_RIS'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_ris_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)


def runRISExpAna(name, args):
    if args.iid:
        isIID = 'iid'
    else:
        isIID = 'niid'
    if args.local_ep == 5:
        descFun = 'minibatch'
    else:
        descFun = 'fullbatch'

    method = 'Proposed_RIS'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_ris/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_ll_{args.LL}_ris_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_ris/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)


def runExpVarify(name, args):
    if args.iid:
        isIID = 'iid'
    else:
        isIID = 'niid'
    if args.local_bs != 0:
        descFun = 'minibatch'
    else:
        descFun = 'fullbatch'

    method = 'Noiseless'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)

    method = 'OFDMA'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_05_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)

    method = 'Ori_AS'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)

    method = 'Ori_ML'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)

    method = 'Proposed_SCA'
    final_result, net_glob = Main(args, method)  # 主函数
    filename = f'./store_10_4/{name}/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_{args.dataset}_{isIID}.npz'
    np.savez(filename, vars(args), final_result)
    modelFileName = f'./store_10_4/{name}/{method}_{args.dataset}_{descFun}_{isIID}.pth'
    torch.save(net_glob, modelFileName)


if __name__ == '__main__':
    startTime = time.time()
    print(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(cp.installed_solvers())

    # proposed 设置：
    args.vv = '1e-7'
    args.cur = 1
    args.count = 0

    # AS 设置：
    args.ASAdjust = False

    # 通用
    # args.dataset = 'mnist'
    # method = 'Ori_ML'

    # ---------------- mnist -----------------------------------------
    # 这个跑需要保存模型画ROC，用了新的参数
    # name = set_mnist_minibatch_iid(args)
    # runExp(name, args)
    # name = set_mnist_minibatch_niid(args)
    # runExp(name, args)
    # name = set_mnist_fullbatch_iid(args)
    # runExp(name, args)
    # name = set_mnist_fullbatch_niid(args)
    # runExp(name, args)

    # ------------ cifar -----------------------------------------
    # name = set_cifar_minibatch_iid(args)
    # runExp(name, args)
    # name = set_cifar_minibatch_niid(args)
    # runExp(name, args)

    # ------------ varify -----------------------------------------
    # MNIST 验证 B=300，E=1（5 * 300 = 60000 / 40），注意不要覆盖原始fig文件
    # lrList = [0.025, 0.030, 0.035]
    # for lr in lrList:
    #     name = set_mnist_minibatch_iid(args)
    #     args.local_ep = 5
    #     args.local_bs = 300
    #     args.lr = lr
    #     runExpVarify(name, args)
    #
    #     name = set_mnist_minibatch_niid(args)
    #     args.local_ep = 5
    #     args.local_bs = 300
    #     args.lr = lr
    #     runExpVarify(name, args)

    # TODO: CIFAR-10 验证 B=128，E=15（原本就是这样），注意不要覆盖原始fig文件
    name = set_cifar_minibatch_iid(args)
    args.local_ep = 5
    args.local_bs = 128
    args.lr = 0.05
    runExpVarify(name, args)

    name = set_cifar_minibatch_niid(args)
    args.local_ep = 5
    args.local_bs = 128
    args.lr = 0.05
    runExpVarify(name, args)

    # ------------ RIS mnist -----------------------------------------
    # name = set_RIS_mnist_minibatch_iid(args)
    # runRISExp(name, args)
    #
    # name = set_RIS_mnist_minibatch_niid(args)
    # runRISExp(name, args)

    # ------------ RIS fmnist -----------------------------------------
    # name = set_RIS_fmnist_minibatch_iid(args)
    # runRISExp(name, args)
    # 改成-18dB再跑一次，正常是-16dB
    # name = set_RIS_fmnist_minibatch_niid(args)
    # args.EsN0dB = -20.0
    # runRISExp(name, args)

    # ------------ RIS EsN0 -----------------------------------------
    # name = set_RIS_EsN0(args)
    # EsN0dB_list_ris = [-15.0]
    # for e in EsN0dB_list_ris:
    #     args.EsN0dB = e
    #     runRISExp(name, args)

    # ------------ RIS CIFAR -----------------------------------------
    # name = set_RIS_cifar_minibatch_iid(args)
    # runRISExp(name, args)
    # name = set_RIS_cifar_minibatch_niid(args)
    # runRISExp(name, args)

    # ------------ RIS 天线 -----------------------------------------
    # LLList = [8]
    # for l in LLList:
    #     args.LL = l
    #     name = set_RIS_ana_mnist_minibatch_iid(args)
    #     runRISExpAna(name, args)

    # Record Running Time
    print(f"end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"time consume: {time.time() - startTime} s")

    # args.EsN0dB = -16.0
    # args.epochs = 200
    # args.lr = 0.1
    # args.iid = False
    # args.local_ep = 5
    # args.local_bs = 0
    #
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # ---------------- Mnist: mini-batch: ep=5, bs=0, lr=0.1, epochs=200
    # args.EsN0dB = -20.0
    # args.epochs = 200
    # args.lr = 0.1
    # args.iid = True
    # args.local_ep = 5
    # args.local_bs = 0
    # method = 'Noiseless'
    # final_result, net_glob = Main(args, method)  # 主函数
    # filename = f'./store_10_4/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)
    # ---------------- Mnist: full-batch: ep=1, bs=0, lr=0.1, epochs=300
    # args.EsN0dB = -26.0
    # args.epochs = 300
    # args.lr = 0.1
    # args.vv = '1e-7'
    # args.iid = True
    # args.local_ep = 1
    # args.local_bs = 0
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_iid.npz'
    # np.savez(filename, vars(args), final_result)
    #
    # args.EsN0dB = -20.0
    # args.epochs = 300
    # args.lr = 0.1
    # args.vv = '1e-7'
    # args.iid = False
    # args.local_ep = 1
    # args.local_bs = 0
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # ------------------------- EsN0：也是按照以前的 E=5 B=128 -----------------------
    # EsN0dB_list_02 = [-26.0, -27.0, -28.0, -29.0]
    # for i in EsN0dB_list_02:
    #     args.EsN0dB = i
    #     args.epochs = 100
    #     args.lr = 0.01
    #     args.iid = True
    #     args.local_ep = 5
    #     args.local_bs = 128
    #     final_result = Main(args, method)  # 主函数
    #     filename = f'./store_final/EsN0/02/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_iid.npz'
    #     np.savez(filename, vars(args), final_result)

    # c
    # args.EsN0dB = -8.0
    # args.epochs = 100
    # args.lr = 0.05
    # args.iid = True
    # args.local_ep = 5
    # args.local_bs = 128
    #
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_cifar_iid.npz'
    # np.savez(filename, vars(args), final_result)
    #
    # args.EsN0dB = -10.0
    # args.epochs = 100
    # args.lr = 0.05
    # args.iid = False
    # args.local_ep = 5
    # args.local_bs = 128
    #
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_cifar_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # args.test = 3
    # 不行试试cvxpy求解器
    # cifar只跑一下as和ml即可，因为这两个地方有变化，按照原来的E=5，B=128
    # N vs acc：只涉及到了1，按照以前的E=5，B=128
    # EsN0：也是按照以前的E=5，B=128，重新跑02和04即可

    # method = 'Noiseless'
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_mnist_iid.npz'
    # np.savez(filename, vars(args), final_result)

    # method = 'OFDMA'
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_05_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # method = 'OAC'
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_06_mnist_iid.npz'
    # np.savez(filename, vars(args), final_result)

    # method = 'Proposed_SCA'
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # method = 'Ori_AS'
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # method = 'Ori_ML'
    # final_result = Main(args, method)  # 主函数
    # filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_niid.npz'
    # np.savez(filename, vars(args), final_result)

    # EsN0dB_list = [-26.0, -27.0, -28.0, -29.0, -30.0, -31.0, -32.0, -33.0]
    # for i in EsN0dB_list:
    #     args.EsN0dB = i
    #     args.epochs = 150
    #     args.lr = 0.1
    #     args.iid = True
    #     args.dataset = 'mnist'
    #     args.local_ep = 5
    #     args.local_bs = 1500
    #     args.cur = 1
    #     args.count = 0
    #
    #     method = 'Proposed_SCA'
    #     final_result = Main(args, method)  # 主函数
    #     filename = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01.npz'
    #     np.savez(filename, vars(args), final_result)

    # 天线
    # N_list = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    # N_list = [20, 12, 6]
    # for i in N_list:
    #     args.EsN0dB = -28.0
    #     args.N = i
    #     args.epochs = 100
    #     args.lr = 0.01
    #     args.frac = 0.2
    #     args.local_ep = 5
    #     args.local_bs = 128
    #     method = 'Proposed_SCA'
    #     final_result = Main(args, method)  # 主函数
    #     filename = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_ddd.npz'
    #     np.savez(filename, vars(args), final_result)

    # EsN0dB_list = [20.0, 10.0, 0.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0]
    # for i in EsN0dB_list:
    #     args.EsN0dB = i
    #     args.epochs = 100
    #     args.lr = 0.01
    #     args.iid = True
    #
    #     method = 'Ori_AS'
    #     final_result = Main(args, method)  # 主函数
    #     filename = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_D1.npz'
    #     np.savez(filename, vars(args), final_result)

    # 重新测试mnist-iid中AS
    # EsN0dB_list = [-26.0, -25.8, -25.6, -25.4]
    # for i in EsN0dB_list:
    #     args.EsN0dB = i
    #     args.epochs = 300
    #     args.lr = 0.1
    #     args.vv = '1e-7'
    #     args.iid = True
    #     args.dataset = 'mnist'
    #     # args.dataset = 'cifar'
    #     args.local_ep = 1
    #     args.cur = 1
    #     args.count = 0
    #     args.ASAdjust = False
    #
    #     method = 'Ori_AS'
    #     final_result = Main(args, method)  # 主函数
    #     filename = f'./store_fullbatch/new04/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_D1.npz'
    #     np.savez(filename, vars(args), final_result)
