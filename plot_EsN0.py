"""
作者：hp
日期：2022年11月07日
"""
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import copy
from utils.options import args_parser


def pl(result_list, EsN0dB):
    result1_x = result_list[0]
    result1_y = np.array(result_list[1])
    result2_x = result_list[2]
    result2_y = np.array(result_list[3])
    result3_x = result_list[4]
    result3_y = np.array(result_list[5])
    result4_x = result_list[6]
    result4_y = np.array(result_list[7])

    # plt.rcParams["font.family"] = "SimHei"
    plt.figure('Proposed_test_acc')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.title(f'Proposed_test_acc_SNR={EsN0dB}')

    plt.plot(result3_x, result3_y / 100.,
             label=r'Proposed SCA-based Algorithm', color='#1f77b4', marker='o', markevery=1, markersize=5,
             markerfacecolor='none')
    plt.plot(result1_x, result1_y / 100.,
             label=r'ML Estimator', color='#2ca02c', marker='*', markevery=1, markersize=8)
    plt.plot(result2_x, result2_y / 100.,
             label=r'Aligned-Sample Estimator', color='#d62728', marker='s', markevery=1, markersize=5,
             markerfacecolor='none')
    plt.plot(result4_x, result4_y / 100.,
             label=r'Aligned ML Estimator', color='#ff7f0e', marker='s', markevery=1, markersize=5,
             markerfacecolor='none')

    label_font = {
        'family': 'Arial',  # 字体
        'style': 'normal',
        'size': 14,  # 字号
        'weight': "normal",  # 是否加粗，不加粗
    }
    plt.ylabel('Test Accuracy', fontdict=label_font)
    plt.xlabel('EsN0(dB)', fontdict=label_font)
    legend_font = {
        'family': 'Arial',  # 字体
        'style': 'normal',
        'size': 10,  # 字号
        'weight': "normal",  # 是否加粗，不加粗
    }
    plt.legend(frameon=True, prop=legend_font)
    # plt.xlim((0, 100))
    # plt.ylim((0, 1))
    plt.yticks(np.linspace(0, 1, 11), size=12)
    plt.xticks(size=12)
    plt.grid()
    plt.show(block=True)


if __name__ == '__main__':
    args = args_parser()
    # -------------------------------------------新版在：MisAlignedOAC_11_3 - 副本-------------------------------

    EsN0dB_list_01 = [10.0, 0.0, -10.0, -14.0, -16.0, -18.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0, -26.0, -27.0]
    acc_list_01 = []
    args.lr = 0.01
    args.epochs = 100
    for EsN0dB in EsN0dB_list_01:
        filename = f'./store_final/EsN0/01/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01.npz'
        result_set = np.load(filename, allow_pickle=True)['arr_1'].tolist()
        acc_list_01.append(np.mean(result_set['test_acc'][-20:]))
    EsN0dB_list_01.append(-28.0)
    acc_list_01.append(np.mean(np.array([10.65, 12.58, 10.64, 7.04, 14.41, 10.27, 9.16], dtype=float)))

    EsN0dB_list_02 = [50.0, 40.0, 35.0, 30.0, 25.0, 24.0, 23.0, 22.0, 21.9, 21.5, 21.4, 21.2, 21.1, 21.0]
    # EsN0dB_list_02 = [50.0, 40.0, 35.0, 30.0, 25.0, 24.0, 23.0, 22.0, 21.9, 21.5, 21.4, 21.2, 21.1, 21.0]
    acc_list_02 = []
    args.lr = 0.01
    args.epochs = 100
    for EsN0dB in EsN0dB_list_02:
        filename = f'./store_final/EsN0/02/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_plusP.npz'
        result_set = np.load(filename, allow_pickle=True)['arr_1'].tolist()
        acc_list_02.append(np.mean(result_set['test_acc'][-20:]))

    EsN0dB_list_02_2 = [20.0, 10.0, 0.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0,
                        -17.0, -18.0, -19.0, -20.0, -22.0, -23.0, -25.0, -27.0, -28.0]
    acc_list_02_2 = []
    args.lr = 0.01
    args.epochs = 100
    for EsN0dB in EsN0dB_list_02_2:
        filename = f'./store_final/EsN0/02/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_02_mnist_iid.npz'
        result_set = np.load(filename, allow_pickle=True)['arr_1'].tolist()
        acc_list_02_2.append(np.mean(result_set['test_acc'][-20:]))

    EsN0dB_list_04 = [20.0, 10.0, 0.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0]
    acc_list_04 = []
    args.lr = 0.01
    args.epochs = 100
    for EsN0dB in EsN0dB_list_04:
        filename = f'./store_final/EsN0/04/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_04_D1.npz'
        result_set = np.load(filename, allow_pickle=True)['arr_1'].tolist()
        acc_list_04.append(np.mean(result_set['test_acc'][-20:]))

    # filename1 = f'./store_final/minist/_M_{args.M_Prime}_N_{args.N}_EsN0dB_{args.EsN0dB}_epoch_{args.epochs}_lr_{args.lr}_01.npz'

    result_list = [
        EsN0dB_list_02,
        acc_list_02,
        EsN0dB_list_04,
        acc_list_04,
        EsN0dB_list_01,
        acc_list_01,
        EsN0dB_list_02_2,
        acc_list_02_2
    ]

    pl(result_list, args.EsN0dB)

    # print('a')
