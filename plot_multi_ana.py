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

    plt.figure('Proposed_test_acc')
    # plt.title(f'Proposed_test_acc_SNR={EsN0dB}')
    plt.plot(range(len(result1['test_acc'])), result1['test_acc'] / 100.,
             label=r'N=20', color='#1f77b4', marker='o', markevery=5, markersize=5,
             markerfacecolor='none')
    plt.plot(range(len(result2['test_acc'])), result2['test_acc'] / 100.,
             label=r'N=12', color='#ff7f0e', marker='s', markevery=5, markersize=5, markerfacecolor='none')
    plt.plot(range(len(result3['test_acc'])), result3['test_acc'] / 100.,
             label=r'N=6', color='#2ca02c', marker='*', markevery=5, markersize=8)
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
    plt.xlim((0, 100))
    plt.ylim((0, 1))
    plt.yticks(np.linspace(0, 1, 11), size=12)
    plt.xticks(size=12)
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

    args.EsN0dB = -29.0
    args.lr = 0.01
    args.epochs = 100

    filename1 = f'./store/_M_{args.M_Prime}_N_{20}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_ddd.npz'
    filename2 = f'./store/_M_{args.M_Prime}_N_{12}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_ddd.npz'
    filename3 = f'./store/_M_{args.M_Prime}_N_{6}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01_ddd.npz'

    # filename = './store_2/save/epc_100_label_2_3.npz'  # 2 优秀 ，3 优秀

    a = np.load(filename1, allow_pickle=True)
    b = np.load(filename2, allow_pickle=True)
    c = np.load(filename3, allow_pickle=True)

    libopt_dic = a['arr_0'].tolist()
    result_set1 = a['arr_1'].tolist()
    result_set2 = b['arr_1'].tolist()
    result_set3 = c['arr_1'].tolist()

    result_list = [
        result_set1,
        result_set2,
        result_set3,
    ]

    pl(result_list, args.EsN0dB)

    # print('a')
