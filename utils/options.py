#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--short', type=float, default=0, help="short or long pkt")
    parser.add_argument('--maxDelay', type=float, default=0.9, help="the maximum delay of the devices")
    parser.add_argument('--T_sam', type=float, default=1.0, help="the sample period")
    parser.add_argument('--phaseOffset', type=float, default=1, help="phase offsets, can be 0->0; 1->2pi/4; 2->2pi/2; 3->2pi")
    # parser.add_argument('--phaseOffset', type=float, default=0, help="phase offsets, can be 0->0; 1->2pi/4; 2->2pi/2; 3->2pi")
    parser.add_argument('--EsN0dB', type=float, default=45.0, help="variance of the noise")  # 后面改成SNR
    # parser.add_argument('--Estimator', type=float, default=1, help="1->aligned_sample,2->LMMSE")
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--P0', type=int, default=10, help="Maximum transmit power")
    parser.add_argument('--M_Prime', type=int, default=40, help="number of users: M_Prime")
    parser.add_argument('--N', type=int, default=5, help="number of antennas: N")
    parser.add_argument('--L', type=int, default=260, help="length of one packet: L")
    parser.add_argument('--LL', type=int, default=10, help="RIS unit LL")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=1500, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    # parser.add_argument('--lr', type=float, default=0.001, help="learning rate")  # 测试non-iid
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")  # iid
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--epsilon', type=float, default=1e-2, help="SCA iteration stop condition")
    parser.add_argument('--epsilon_RIS', type=float, default=1e-3, help="RIS iteration stop condition")
    parser.add_argument('--epsilon_RIS_forff', type=float, default=1e-3, help="RIS iteration stop condition forff")
    parser.add_argument('--gap_gap_forff', type=float, default=1e-4, help="RIS iteration stop condition forff early stop")
    parser.add_argument('--epsilon_RIS_forTheta', type=float, default=1e-4, help="RIS iteration stop condition forTheta")
    parser.add_argument('--gap_gap_forTheta', type=float, default=1e-5, help="RIS iteration stop condition forTheta early stop")
    parser.add_argument('--SCA_I_max', type=int, default=100, help="SCA maximum Iterations")
    parser.add_argument('--RIS_I_max', type=int, default=100, help="RIS maximum Iterations")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--batch_norm', type=bool, default=False, help="batch_norm or dropout")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    # parser.add_argument('--dataset', type=str, default='fmnist', help="name of dataset")
    # parser.add_argument('--iid', action='store_true', default=False, help='whether i.i.d or not')
    parser.add_argument('--iid', action='store_true', default=True, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', default=1, action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
