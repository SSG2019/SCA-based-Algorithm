"""
作者：hp
日期：2022年08月26日
"""
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import copy
from utils.options import args_parser


def pl(result_list, EsN0dB):
    result1 = result_list[0]
    result2 = result_list[1]
    result3 = result_list[2]
    result4 = result_list[3]
    result5 = result_list[4]

    # plt.rcParams["font.family"] = "SimHei"
    # plt.figure('Proposed_train_loss')
    # # plt.title(f'Proposed_train_loss_SNR={EsN0dB}')
    # plt.plot(range(len(result1['train_loss'])), result1['train_loss'], label=r'Proposed SCA-based Algorithm')
    # plt.plot(range(len(result2['train_loss'])), result2['train_loss'], label=r'ML Estimator')
    # plt.plot(range(len(result3['train_loss'])), result3['train_loss'], label=r'Aligned-Noiseless')
    # plt.plot(range(len(result4['train_loss'])), result4['train_loss'], label=r'Aligned-Sample Estimator')
    # plt.ylabel('Proposed_train_loss', fontsize=10)
    # plt.xlabel('Communication round', fontsize=10)
    # plt.legend()
    # # plt.xlim((0, 1000))
    # plt.xlim()
    # plt.grid()
    # plt.show(block=True)
    plt.figure('Proposed_test_acc')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.title(f'Proposed_test_acc_SNR={EsN0dB}')

    markevery = 10
    markersize = 6
    plt.plot(range(len(result1['test_acc'])), result1['test_acc'] / 100.,
             label=r'Proposed SCA-based Algorithm', color='#1f77b4', marker='o', markevery=markevery,
             markersize=markersize,
             markerfacecolor='none')
    plt.plot(range(len(result3['test_acc'])), result3['test_acc'] / 100.,
             label=r'Aligned-Noiseless Aggregation', color='#2ca02c', linestyle='--')
    plt.plot(range(len(result2['test_acc'])), result2['test_acc'] / 100.,
             label=r'ML Estimator', color='#ff7f0e', marker='*', markevery=markevery, markersize=8)
    plt.plot(range(len(result4['test_acc'])), result4['test_acc'] / 100.,
             label=r'Aligned-Sample Estimator', color='#d62728', marker='s', markevery=markevery, markersize=markersize,
             markerfacecolor='none')
    plt.plot(range(len(result5['test_acc'])), result5['test_acc'] / 100.,
             label=r'OFDMA', color='black', markevery=markevery, markersize=markersize,
             markerfacecolor='none')
    label_font = {
        'family': 'Arial',  # 字体
        'style': 'normal',
        'size': 14,  # 字号
        'weight': "normal",  # 是否加粗，不加粗
    }
    plt.ylabel('Test Accuracy', fontdict=label_font)
    plt.xlabel('Communication round', fontdict=label_font)
    legend_font = {
        'family': 'Arial',  # 字体
        'style': 'normal',
        'size': 10,  # 字号
        'weight': "normal",  # 是否加粗，不加粗
    }
    plt.legend(frameon=True, prop=legend_font)
    # plt.legend(loc='upper left', frameon=True, prop=legend_font)
    # plt.xlim((0, 400))
    # plt.ylim((0, 1.0))
    # plt.yticks(np.linspace(0, 1, 11), size=12)
    # plt.xticks(size=12)
    plt.grid()
    plt.show(block=True)

    # plt.figure(num3)
    # plt.title(num3)
    # # plt.plot(range(len(result['MNIST_accuracy_test'])), result['MNIST_accuracy_test'], label=r'Noiseless Channel')
    # plt.plot(range(len(result['Task_2_test_accuracy_proposed'])), result['Task_2_test_accuracy_proposed'],
    #          label=r'feature extraction - label classifier')
    # plt.plot(range(len(result['Task_2_test_accuracy_Noencoder'])), result['Task_2_test_accuracy_Noencoder'], '--',
    #          label=r'No feature extraction network')
    # # plt.plot(range(len(result1['MNIST_accuracy_test1'])), result1['MNIST_accuracy_test1'],
    # #          label=r'Random Device Selection')
    # # plt.plot(range(len(result['accuracy_test3'])), result['accuracy_test3'],label=r'Wuthout RIS')
    # # plt.plot(range(len(result['accuracy_test2'])),result['accuracy_test2'],label=r'DC Programming')
    # # plt.plot(range(len(result['accuracy_test5'])), result['accuracy_test5'],label=r'Deffiential Geometry')
    # # plt.ylabel('Task 2 Test Accuracy')
    # # plt.xlabel('Task 2 Training Round')
    # plt.ylabel('Classifier_2_test_acc', fontsize=16)
    # plt.xlabel('Epoch', fontsize=16)
    # plt.legend()
    # # plt.xlim((0, 1000))
    # plt.xlim()
    # plt.grid()
    # plt.show(block=True)

    # plt.figure(num4)
    # plt.title(num4)
    # # plt.plot(range(len(result['MNIST_loss_train'])), result['MNIST_loss_train'], label=r'Noiseless Channel')
    # plt.plot(range(len(result['Task_2_train_loss_proposed'])), result['Task_2_train_loss_proposed'],
    #          label=r'feature extraction - label classifier')
    # plt.plot(range(len(result['Task_2_train_loss_Noencoder'])), result['Task_2_train_loss_Noencoder'], '--',
    #          label=r'No feature extraction network')
    # # plt.plot(range(len(result1['MNIST_loss_train1'])), result1['MNIST_loss_train1'], label=r'Random Device Selection')
    # # plt.plot(range(len(result['accuracy_test3'])), result['accuracy_test3'],label=r'Wuthout RIS')
    # # plt.plot(range(len(result['accuracy_test2'])),result['accuracy_test2'],label=r'DC Programming')
    # # plt.plot(range(len(result['accuracy_test5'])), result['accuracy_test5'],label=r'Deffiential Geometry')
    # # plt.ylabel('Task 2 Training Loss')
    # # plt.xlabel('Task 2 Training Round')
    # plt.ylabel('Classifier_2_train_loss', fontsize=16)
    # plt.xlabel('Epoch', fontsize=16)
    # plt.legend()
    # # plt.xlim((0, 1000))
    # plt.xlim()
    # # plt.ylim((0, 10))
    # plt.grid()
    # plt.show(block=True)

    # plt.ylim([0, 50])
    # len1=len(result['accuracy_test'])
    # a=np.zeros([5,len1])
    # a[0,:]=result['accuracy_test']
    # a[1,:]=result['accuracy_test1']
    # a[2,:]=result['accuracy_test3']
    # a[3,:]=result['accuracy_test2']
    # a[4,:]=result['accuracy_test5']
    # return a


if __name__ == '__main__':
    args = args_parser()
    # -------------------------------------------新版在：MisAlignedOAC_11_3 - 副本-------------------------------

    # minist-iid
    # args.EsN0dB = -20.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01.npz'
    # filename2 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_plusP.npz'
    # filename3 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03.npz'
    # filename4 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_D1.npz'

    # minist-non-iid
    # args.EsN0dB = -18.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_niid.npz'
    # filename2 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_niid.npz'
    # filename3 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_niid.npz'
    # filename4 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_niid.npz'

    # mnist-non-iid new edition
    # filename1 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-16.0}_epoch_{100}_lr_{0.01}_01_minist.npz'
    # filename2 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-16.0}_epoch_{100}_lr_{0.01}_02_minist.npz'
    # filename3 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-16.0}_epoch_{100}_lr_{0.01}_03_minist.npz'
    # filename4 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-16.0}_epoch_{100}_lr_{0.01}_04_minist.npz'

    # cifar-iid
    # args.EsN0dB = -20.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_final/cifar/iid/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01.npz'
    # filename2 = f'./store_final/cifar/iid/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_plusP.npz'
    # filename3 = f'./store_final/cifar/iid/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03.npz'
    # filename4 = f'./store_final/cifar/iid/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_D1.npz'

    # # cifar-non-iid shard （之前按100用户跑的错误数据）
    # args.EsN0dB = -20.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_final/cifar/shard/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_niid.npz'
    # filename2 = f'./store_final/cifar/shard/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_niid.npz'
    # filename3 = f'./store_final/cifar/shard/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_niid.npz'
    # filename4 = f'./store_final/cifar/shard/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_niid.npz'

    # cifar-non-iid shard 40000
    # args.EsN0dB = -18.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_final/cifar/shard40000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_niid.npz'
    # filename2 = f'./store_final/cifar/shard40000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_niid.npz'
    # filename3 = f'./store_final/cifar/shard40000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_niid.npz'
    # filename4 = f'./store_final/cifar/shard40000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_niid.npz'

    # cifar-non-iid shard 30000
    # args.EsN0dB = -20.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_final/cifar/shard30000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_niid.npz'
    # filename2 = f'./store_final/cifar/shard30000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_niid.npz'
    # filename3 = f'./store_final/cifar/shard30000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_03_niid.npz'
    # filename4 = f'./store_final/cifar/shard30000/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_niid.npz'

    # # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{90.0}_epoch_{args.epochs}_lr_{args.lr}_01.npz'
    # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{args.epochs}_lr_{args.lr}_01.npz'
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{90.0}_epoch_{args.epochs}_lr_{args.lr}_03.npz'
    # # cifar edition
    # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{90.0}_epoch_{3}_lr_{0.05}_01_cifar.npz'
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{90.0}_epoch_{3}_lr_{0.05}_03_cifar.npz'

    # -20.0db 效果较差
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{100}_lr_{0.05}_03_cifar.npz'

    # 测试0.01 noniid
    # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-16.0}_epoch_{100}_lr_{0.01}_03_cifar_noniid.npz'
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-16.0}_epoch_{100}_lr_{0.01}_03_cifar_noniid.npz'

    # cifar iid -12.0db
    # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-12.0}_epoch_{100}_lr_{0.05}_01_cifar.npz'
    # filename2 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-12.0}_epoch_{100}_lr_{0.05}_02_cifar.npz'
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-12.0}_epoch_{100}_lr_{0.05}_03_cifar.npz'
    # filename4 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-12.0}_epoch_{100}_lr_{0.05}_04_cifar.npz'
    # cifar iid -10.0db
    # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_01_cifar.npz'
    # filename2 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_02_cifar.npz'
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_03_cifar.npz'
    # filename4 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_04_cifar.npz'
    # cifar iid -8.0db
    # filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_01_cifar.npz'
    # filename2 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_02_cifar.npz'
    # filename3 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_03_cifar.npz'
    # filename4 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_04_cifar.npz'

    # cifar non-iid -10.0db \store_final\cifar\shard0
    # filename1 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_01_niid.npz'
    # filename2 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_02_niid.npz'
    # filename3 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_shard_0_epoch_{100}_lr_{0.05}_03_niid.npz'
    # filename4 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_04_niid.npz'

    # Ana-minist-iid E=5, B=128, M=8
    # args.EsN0dB = -29.0
    # filename1 = f'./store_final/Ana/_M_{args.M_Prime}_N_{6}_EsN0dB_{args.EsN0dB}_epoch_{100}_lr_{0.01}_01_ddd.npz'
    # filename2 = f'./store_final/Ana/_M_{args.M_Prime}_N_{12}_EsN0dB_{args.EsN0dB}_epoch_{100}_lr_{0.01}_01_ddd.npz'
    # filename3 = f'./store_final/Ana/_M_{args.M_Prime}_N_{20}_EsN0dB_{args.EsN0dB}_epoch_{100}_lr_{0.01}_01_ddd.npz'
    # filename4 = f'./store_final/Ana/_M_{args.M_Prime}_N_{6}_EsN0dB_{args.EsN0dB}_epoch_{100}_lr_{0.01}_01_ddd.npz'

    # 2023/05: new ===================================================================================================

    # ============ mnist-iid E=5, B=0, AS=False ============
    # args.EsN0dB = -20.0
    # args.lr = 0.1
    # args.epochs = 200
    # filename1 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_mnist_iid.npz'
    # filename2 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_iid.npz'
    # filename3 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{args.epochs}_lr_{args.lr}_03_mnist_iid.npz'
    # filename4 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_mnist_iid.npz'

    # ============ mnist-niid E=5, B=0, AS=False ============
    # args.EsN0dB = -16.0
    # args.lr = 0.1
    # args.epochs = 200
    # filename1 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_mnist_niid.npz'
    # filename2 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_niid.npz'
    # filename3 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-18.0}_epoch_{args.epochs}_lr_{args.lr}_03_mnist_niid.npz'
    # # filename3 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{100}_lr_{args.lr}_03_cifar_iid.npz'
    # # filename3 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{400}_lr_{args.lr}_03_mnist_iid.npz'
    # filename4 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_mnist_niid.npz'

    # =========================== full-batch and E = 0 ==============================
    # ============ mnist-iid E=1, B=0 ，对AS还是正常取-26就行，只不过截取230之前的round
    # args.EsN0dB = -26.0
    # args.lr = 0.1
    # args.epochs = 300
    # filename1 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_mnist_iid.npz'
    # filename2 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_iid.npz'
    # # filename2 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{200}_lr_{args.lr}_02_mnist_iid.npz'
    # filename3 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{300}_lr_{args.lr}_03_mnist_iid.npz'
    # filename4 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-26.0}_epoch_{args.epochs}_lr_{args.lr}_04_mnist_iid.npz'
    # # 重新跑的AS：
    # filename4 = f'./store_fullbatch/new04/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-25.8}_epoch_{args.epochs}_lr_{args.lr}_04_D1.npz'

    # ============ mnist-niid E=1, B=0 ============
    # 需要局部放大
    # args.EsN0dB = -20.0
    # args.lr = 0.1
    # args.epochs = 300
    # filename1 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_mnist_niid.npz'
    # filename2 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_niid.npz'
    # # filename2 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{200}_lr_{args.lr}_02_mnist_iid.npz'
    # filename3 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-26.0}_epoch_{300}_lr_{args.lr}_03_mnist_niid.npz'
    # filename4 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{args.epochs}_lr_{args.lr}_04_mnist_niid.npz'

    # # 测试02的Esn0
    # filename2 = f'./store_fullbatch/EsN0/02/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{21.2}_epoch_{100}_lr_{0.01}_02.npz'

    # =================== CIFAR：iid
    # cifar iid -8.0db
    filename1 = f'./store/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_01_cifar.npz'
    filename2 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_02_cifar_iid.npz'
    filename3 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_03_cifar_iid.npz'
    filename4 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_04_cifar_iid.npz'
    filename5 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-8.0}_epoch_{100}_lr_{0.05}_05_cifar_iid.npz'

    # =================== CIFAR：non-iid
    # cifar non-iid -10.0db \store_final\cifar\shard0
    # filename1 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_01_niid.npz'
    # # filename2 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_02_niid.npz'
    # filename2 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_02_cifar_niid.npz'
    # # filename3 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_03_cifar_niid.npz'
    # filename3 = f'./store_final/cifar/shard0/_M_{args.M_Prime}_N_{args.N}_shard_0_epoch_{100}_lr_{0.05}_03_niid.npz'
    # filename4 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-11.0}_epoch_{100}_lr_{0.05}_04_cifar_niid.npz'
    # filename5 = f'./store_fullbatch/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.05}_05_cifar_niid.npz'

    # ======================== 测试用 receiver power
    # args.EsN0dB = -6.0
    # args.lr = 0.01
    # args.epochs = 100
    # filename1 = f'./store_fullbatch/newReceiveSNR/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_re.npz'
    # filename2 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{200}_lr_{0.1}_02_minist_iid.npz'
    # filename3 = f'./store_fullbatch/newReceiveSNR/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-10.0}_epoch_{100}_lr_{0.01}_03_iid.npz'
    # # filename4 = f'./store_fullbatch/newReceiveSNR/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_re.npz'
    # # filename4 = f'./store_fullbatch/newReceiveSNR/True/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_re.npz'
    # filename4 = f'./store_fullbatch/newReceiveSNR/nonAcc_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_re.npz'

    # args.EsN0dB = -10.0
    # args.epochs = 300
    # args.lr = 0.1
    # filename1 = f'./store_fullbatch/newReceiveSNR/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_re.npz'
    # filename2 = f'./store_fullbatch/mnist_1/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{-20.0}_epoch_{200}_lr_{0.1}_02_minist_iid.npz'
    # filename3 = f'./store_fullbatch/newReceiveSNR/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{45.0}_epoch_{300}_lr_{0.1}_03_iid.npz'
    # filename4 = f'./store_fullbatch/newReceiveSNR/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_re.npz'

    a = np.load(filename1, allow_pickle=True)
    b = np.load(filename2, allow_pickle=True)
    c = np.load(filename3, allow_pickle=True)
    d = np.load(filename4, allow_pickle=True)
    e = np.load(filename5, allow_pickle=True)

    libopt_dic = a['arr_0'].tolist()
    result_set1 = a['arr_1'].tolist()
    result_set2 = b['arr_1'].tolist()
    result_set3 = c['arr_1'].tolist()
    result_set4 = d['arr_1'].tolist()
    result_set5 = e['arr_1'].tolist()

    acc1 = np.mean(result_set1['test_acc'][-20:])
    acc2 = np.mean(result_set2['test_acc'][-20:])
    acc3 = np.mean(result_set3['test_acc'][-20:])
    acc4 = np.mean(result_set4['test_acc'][-20:])
    acc5 = np.mean(result_set5['test_acc'][-20:])

    result_list = [
        result_set1,
        result_set2,
        result_set3,
        result_set4,
        result_set5,
    ]

    pl(result_list, args.EsN0dB)

    # print('a')
