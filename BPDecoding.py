import numpy as np
import pdb
import copy

from numpy.linalg import LinAlgError

from joint import optimize, optimizeRIS


def per_pkt_transmission_Proposed(args, L, TransmittedSymbols, Q, P, V, idxs_users, idx_packet):
    # MM：用户数；TransmittedSymbols：每一个包中所有用户的symbol(4, 260)
    # Pass the channel and generate samples at the receiver
    # complex pass the complex channel：每一个用户的某个数据包中的数据乘以相移这个复数
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)

    K = args.K[idxs_users]

    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, [1, len(TransmittedSymbols[0])])
    TransmittedSymbols = TransmittedSymbols * K_ex

    # var = np.var(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])
    # var[var < 1e-3] = 1e-3

    TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    print('Symbols_var_max =', np.max(Symbols_var))
    print('dd_min =', np.min(dd))
    Symbols_var2 = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L
    # if args.vv == '1e-5':
    #     Symbols_var[Symbols_var < 1e-5] = 1e-5  # < 0.00001
    # elif args.vv == '1e-3':
    #     Symbols_var[Symbols_var < 1e-3] = 1e-3  # < 0.001
    # elif args.vv == '1e-7':
    #     Symbols_var[Symbols_var < 1e-7] = 1e-7
    # else:
    #     print('wrong \n\n\n\n\n\nwrong \n\n\n\n\nwrong')
    # s_bar = (Symbols_mean[np.newaxis, :] @ K).reshape(-1)
    s_bar = np.sum(Symbols_mean)
    N = args.N
    # Symbols_var2 = Symbols_var * 1e5
    obj, ff, pp, cnt = optimize(args, M, N, h, Q, P, dd, Symbols_var2, 0)
    pp = pp * np.sqrt(np.power(10, cnt))

    # 仅用来查看：Pm2 * Vm2 <= P0
    pp_ex = np.tile(pp, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols_pp = TransmittedSymbols * pp_ex
    SigPower = np.var(TransmittedSymbols_pp, axis=1)  # 四个发射功率
    SigPower_mean = np.mean(SigPower)  # 9.742
    print('SigPower_mean =', SigPower_mean)
    print(f'idx_packet = {idx_packet} \n')

    if SigPower_mean < 1e-4:
        SigPower_mean = 1e-4

    D_entry = h.T.conj() @ ff * pp
    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]

    SignalPart = np.sum(TransmittedSymbols, 0)
    Receive_SigPower_Per_Symbol = np.sum(np.power(np.abs(SignalPart), 2)) / L  # 186.119
    EsN0 = np.power(10, args.EsN0dB / 10.0)
    Receive_SigPower = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L  # 每个用户的接收功率
    N0 = SigPower_mean / EsN0
    # N0 = Receive_SigPower_Per_Symbol / EsN0
    # N0 = args.P0 / EsN0
    noisePowerVec = N0 / dd
    # SNR = 10 * np.log(args.P0 / noisePowerVec)

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)  # 相当于每一个元素重复4回: ndarray(4, 1040)
    for idx in np.arange(M):  # 见 pad “015”
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)  # (1043,)

    # generate noise
    for n in range(N):
        for idx in np.arange(M):
            noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
                loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
            if idx == 0:
                AWGNnoise = np.array([noise])
            else:
                AWGNnoise = np.r_[AWGNnoise, np.array([noise])]  # ndarray(4, 261)每一行表示一个匹配滤波对于整个数据包的260+1个噪声。
        AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')  # (1, 1044)
        if n == 0:
            AWGNnoise_f_reshape = np.array([AWGNnoise_reshape.flatten()])
        else:
            AWGNnoise_f_reshape = np.r_[AWGNnoise_f_reshape, np.array([AWGNnoise_reshape.flatten()])]
    noise_plus_f = ff.conj().T @ AWGNnoise_f_reshape
    samples = samples + noise_plus_f[0][0:-1]
    # samples = samples + AWGNnoise_reshape[0][0:-1]  # (1043, )

    D = np.zeros((M * (L + 1) - 1, M * L), dtype=complex)  # (1043, 1040)
    for i in range(M * L):
        D[np.arange(M) + i, i] = D_entry[np.mod(i, M)]
    DzVec = np.tile(noisePowerVec, [1, L + 1])  # 按维度1复制1次（不变），按维度2复制261次，得：(1, 1044)
    Dz = np.linalg.norm(ff, 2) ** 2 * np.diag(DzVec[0][:-1])  # (1043, 1043)

    # ------------------------------------- ML
    MUD = np.matmul(D.conj().T, np.linalg.inv(Dz))  # 矩阵求逆
    MUD = np.matmul(MUD, D)  # 矩阵相乘
    MUD = np.matmul(np.linalg.inv(MUD), D.conj().T)
    MUD = np.matmul(MUD, np.linalg.inv(Dz))
    MUD = np.matmul(MUD, np.array([samples]).T)  # np.array([samples])维度为：(1, 1043)；MUD：(1040, 1)

    # ------------------------------------- Estimate SUM
    output = (V @ MUD).reshape(-1)
    # output = np.sum(np.reshape(MUD, [L, M]), 1)  # (260,)
    output = output + s_bar
    output = output * args.lr
    # output = output / len(K)
    return output


def per_pkt_transmission_Ori_ML(args, L, TransmittedSymbols, V, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    K = args.K[idxs_users]
    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols = TransmittedSymbols * K_ex

    # var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # var[var < 1e-3] = 1e-3
    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])

    # 减去mean
    TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
    # Symbols_mean2 = np.mean(TransmittedSymbols, axis=1)  # 0
    # 此时均值为零，计算方差，即为功率
    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # if args.vv == '1e-5':
    #     Symbols_var[Symbols_var < 1e-5] = 1e-5  # < 0.00001
    # elif args.vv == '1e-3':
    #     Symbols_var[Symbols_var < 1e-3] = 1e-3  # < 0.001
    # elif args.vv == '1e-7':
    #     Symbols_var[Symbols_var < 1e-7] = 1e-7  # < 0.001
    # else:
    #     print('wrong \n\n\n\n\n\nwrong \n\n\n\n\nwrong')
    # s_bar = (Symbols_mean[np.newaxis, :] @ K).reshape(-1)
    s_bar = np.sum(Symbols_mean)

    # 将发射功率调整至最大功率
    # Power_adjust = np.sqrt(args.P0 / Symbols_var)
    # 将发射功率调整至accurate pre-coding
    # Power_adjust = 1 / h[0, :]
    # 不调整
    # Power_adjust[:] = 1.

    # adjust_ex = Power_adjust[:, np.newaxis]
    # adjust_ex = np.tile(adjust_ex, (1, len(TransmittedSymbols[0])))
    # TransmittedSymbols_pp = TransmittedSymbols * adjust_ex
    TransmittedSymbols_pp = TransmittedSymbols
    SigPower = np.var(TransmittedSymbols_pp, axis=1)
    # SigPower2 = np.sum(np.power(np.abs(TransmittedSymbols_pp), 2), axis=1) / L  # 与SigPower相等
    SigPower_mean = np.mean(SigPower)  # 0.00056

    # D_entry = h[0, :] / abs(h[0, :])
    # D_entry = h[0, :]
    D_entry = np.ones(M, dtype=complex)

    # a = abs(D_entry)
    # aaa = abs(D_entry)
    # b = D_entry / abs(D_entry)
    # c = abs(b)

    # D_entry = h[0, :] * Power_adjust

    D_entry = D_entry[:, np.newaxis]
    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]

    SignalPart = np.sum(TransmittedSymbols, 0)
    Receive_SigPower_Per_Symbol = np.sum(np.power(np.abs(SignalPart), 2)) / L  # 1.68
    EsN0 = np.power(10, args.EsN0dB / 10.0)
    N0 = SigPower_mean / EsN0
    # N0 = Receive_SigPower_Per_Symbol / EsN0
    Receive_SigPower = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L  # 每个用户的接收功率（带上h）
    noisePowerVec = N0 / dd

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)  # 相当于每一个元素重复4回: ndarray(4, 1040)
    for idx in np.arange(M):  # 见 pad “015”
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)  # (1043,)

    # generate noise
    for idx in np.arange(M):
        noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
            loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]  # ndarray(4, 261)

    AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')  # (1, 1044)
    samples = samples + AWGNnoise_reshape[0][0:-1]  # (1043, )

    D = np.zeros((M * (L + 1) - 1, M * L), dtype=complex)  # (1043, 1040)
    for i in range(M * L):
        D[np.arange(M) + i, i] = D_entry[np.mod(i, M)]
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = np.diag(DzVec[0][:-1])  # (1043, 1043)
    # a = np.diag(np.ones(Dz.shape[0], dtype=float))

    # ------------------------------------- ML
    MUD = np.matmul(D.conj().T, getInv(Dz))  # 矩阵求逆
    MUD = np.matmul(MUD, D)  # 矩阵相乘
    MUD = np.matmul(getInv(MUD), D.conj().T)
    MUD = np.matmul(MUD, getInv(Dz))
    MUD = np.matmul(MUD, np.array([samples]).T)  # MUD：(1040, 1)

    # ------------------------------------- Estimate SUM
    output = (V @ MUD).reshape(-1)
    output = output + s_bar
    output = output * args.lr
    # output = output / len(K)
    return output


def getInv(Dz):
    try:
        MUD = np.linalg.inv(Dz)  # 矩阵求逆
    except LinAlgError:
        print(Dz)
        a = np.diag(np.ones(Dz.shape[0], dtype=float)) * 1e-5
        return getInv((Dz + a) * 10)
    return MUD


def per_pkt_transmission_Ori_AS(args, L, TransmittedSymbols, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    K = args.K[idxs_users]
    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols = TransmittedSymbols * K_ex

    # var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # var[var < 1e-3] = 1e-3
    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])
    TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # if args.vv == '1e-5':
    #     Symbols_var[Symbols_var < 1e-5] = 1e-5  # < 0.00001
    # elif args.vv == '1e-3':
    #     Symbols_var[Symbols_var < 1e-3] = 1e-3  # < 0.001
    # elif args.vv == '1e-7':
    #     Symbols_var[Symbols_var < 1e-7] = 1e-7  # < 0.001
    # else:
    #     print('wrong \n\n\n\n\n\nwrong \n\n\n\n\nwrong')
    # s_bar = (Symbols_mean[np.newaxis, :] @ K).reshape(-1)
    s_bar = np.sum(Symbols_mean)

    if args.ASAdjust:
        # 手动调整发射功率，使得剩余信道为1
        Power_adjust = 1 / h[0, :]
        adjust_ex = Power_adjust[:, np.newaxis]
        adjust_ex = np.tile(adjust_ex, (1, len(TransmittedSymbols[0])))
        # 通过信道预均衡，手动调整发射功率
        TransmittedSymbols_pp = TransmittedSymbols * adjust_ex
        D_entry = h[0, :] * Power_adjust
    else:
        # 信道直接取1
        TransmittedSymbols_pp = TransmittedSymbols
        D_entry = np.ones(M, dtype=complex)

    SigPower = np.var(TransmittedSymbols_pp, axis=1)
    # SigPower2 = np.sum(np.power(np.abs(TransmittedSymbols_pp), 2), axis=1) / L
    SigPower_mean = np.mean(SigPower)  # 4.613766
    D_entry = D_entry[:, np.newaxis]

    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]

    SignalPart = np.sum(TransmittedSymbols, 0)
    Receive_SigPower_Per_Symbol = np.sum(np.power(np.abs(SignalPart), 2)) / L  # 8.9995
    Receive_SigPower = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L  # 每个用户的接收功率

    EsN0 = np.power(10, args.EsN0dB / 10.0)
    N0 = SigPower_mean / EsN0
    # N0 = Receive_SigPower_Per_Symbol / EsN0

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)  # 相当于每一个元素重复4回: ndarray(4, 1040)
    for idx in np.arange(M):  # 见 pad “015”
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)  # (1043,)

    # generate noise
    for idx in np.arange(M):
        noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
            loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]  # ndarray(4, 261)

    # NoisePower = np.var(AWGNnoise, axis=1)
    # meanNoisePower = np.mean(NoisePower)
    # NoisePower2 = np.sum(np.power(np.abs(AWGNnoise), 2), axis=1) / (L+1)

    AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')  # (1, 1044)
    samples = samples + AWGNnoise_reshape[0][0:-1]  # (1043, )

    MthFiltersIndex = (np.arange(L) + 1) * M - 1
    output = samples[MthFiltersIndex]  # (260,)
    output = output + s_bar
    output = output * args.lr
    # output = output / len(K)
    return output


def per_pkt_transmission_OFDMA(args, L, TransmittedSymbols, V, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    h2 = h[0, :]
    h2[abs(h2) ** 2 < 1e-4] = 1e-2
    K = args.K[idxs_users]
    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols = TransmittedSymbols * K_ex

    # var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # var[var < 1e-3] = 1e-3
    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])

    # 减去mean
    TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
    # Symbols_mean2 = np.mean(TransmittedSymbols, axis=1)  # 0
    # 此时均值为零，计算方差，即为功率
    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # if args.vv == '1e-5':
    #     Symbols_var[Symbols_var < 1e-5] = 1e-5  # < 0.00001
    # elif args.vv == '1e-3':
    #     Symbols_var[Symbols_var < 1e-3] = 1e-3  # < 0.001
    # elif args.vv == '1e-7':
    #     Symbols_var[Symbols_var < 1e-7] = 1e-7  # < 0.001
    # else:
    #     print('wrong \n\n\n\n\n\nwrong \n\n\n\n\nwrong')
    # s_bar = (Symbols_mean[np.newaxis, :] @ K).reshape(-1)
    s_bar = np.sum(Symbols_mean)

    # 将发射功率调整至抵消h2
    Power_adjust = np.sqrt(1 / h2)
    # 不调整
    # Power_adjust[:] = 1.

    adjust_ex = Power_adjust[:, np.newaxis]
    adjust_ex = np.tile(adjust_ex, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols_pp = TransmittedSymbols * adjust_ex
    # TransmittedSymbols_pp = TransmittedSymbols
    SigPowerAdjust = np.var(TransmittedSymbols_pp, axis=1)  # 与Symbols_var相等
    # SigPower2 = np.sum(np.power(np.abs(TransmittedSymbols_pp), 2), axis=1) / L  # 与SigPower相等
    # SigPower_mean = np.mean(Symbols_var)  # 10.0

    EsN0 = np.power(10, args.EsN0dB / 10.0)
    # N0 = SigPower_mean / EsN0
    # noisePower = N0 / 1  # 假设完美同步，采样时间固定为1
    # noisePowerVec = Symbols_var / EsN0
    noisePowerVec = SigPowerAdjust / EsN0
    # n = (np.random.randn(M, L) + 1j * np.random.randn(M, L)) / 2 ** 0.5 * noisePower ** 0.5

    res = np.zeros((M, L), dtype=complex)
    for i in range(M):
        res[i] = TransmittedSymbols[i] + (np.random.randn(1, L) + 1j * np.random.randn(1, L)) / 2 ** 0.5 * \
                 noisePowerVec[i] ** 0.5

    # res = n + TransmittedSymbols
    V2 = np.ones((1, M))

    output = V2 @ res
    output = output + s_bar
    output = output * args.lr
    # output = output / len(K)
    return output


def per_pkt_transmission_OAC(args, L, TransmittedSymbols, V, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    h2 = h[0, :]
    h2[abs(h2) ** 2 < 1e-4] = 1e-2
    K = args.K[idxs_users]
    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols = TransmittedSymbols * K_ex

    # var = np.var(TransmittedSymbols, axis=1)  # (4,)
    # var[var < 1e-3] = 1e-3
    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])

    # 减去mean
    TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
    # Symbols_mean2 = np.mean(TransmittedSymbols, axis=1)  # 0
    # 此时均值为零，计算方差，即为功率
    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    s_bar = np.sum(Symbols_mean)

    # 将发射功率调整至抵消h2
    Power_adjust = np.sqrt(1 / h2)
    # 不调整
    # Power_adjust[:] = 1.

    adjust_ex = Power_adjust[:, np.newaxis]
    adjust_ex = np.tile(adjust_ex, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols_pp = TransmittedSymbols * adjust_ex
    SigPowerAdjust = np.var(TransmittedSymbols_pp, axis=1)  # 与Symbols_var相等

    SigPower_mean = np.mean(SigPowerAdjust)  # 10.0
    # SigPower_mean = np.mean(Symbols_var)  # 10.0
    EsN0 = np.power(10, args.EsN0dB / 10.0)
    # EsN0 = np.power(10, (args.EsN0dB + 6) / 10.0)

    N0 = SigPower_mean / EsN0
    noisePower = N0 / 1  # 假设完美同步，采样时间固定为1
    n = (np.random.randn(M, L) + 1j * np.random.randn(M, L)) / 2 ** 0.5 * noisePower ** 0.5
    res = n + TransmittedSymbols
    V2 = np.ones((1, M))

    output = V2 @ res
    output = output + s_bar
    output = output * args.lr
    # output = output / len(K)
    return output


def per_pkt_transmission_RIS(args, L, TransmittedSymbols, Q, P, V, idxs_users, idx_packet):
    # MM：用户数；TransmittedSymbols：每一个包中所有用户的symbol(4, 260)
    # Pass the channel and generate samples at the receiver
    # complex pass the complex channel：每一个用户的某个数据包中的数据乘以相移这个复数
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    h_DP = copy.deepcopy(h)
    h_RP = 1 / np.sqrt(2) * (np.random.randn(args.N, args.LL) + np.random.randn(args.N, args.LL) * 1j)
    h_DR = 1 / np.sqrt(2) * (np.random.randn(args.LL, M) + np.random.randn(args.LL, M) * 1j)
    G = np.zeros((args.N, args.LL, M), dtype=complex)
    for i in range(M):
        G[:, :, i] = h_RP @ np.diag(h_DR[:, i])

    K = args.K[idxs_users]
    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, [1, len(TransmittedSymbols[0])])
    TransmittedSymbols = TransmittedSymbols * K_ex

    # var = np.var(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])
    # var[var < 1e-3] = 1e-3

    TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    print('Symbols_var_max =', np.max(Symbols_var))
    print('dd_min =', np.min(dd))
    Symbols_var2 = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L
    # if args.vv == '1e-5':
    #     Symbols_var[Symbols_var < 1e-5] = 1e-5  # < 0.00001
    # elif args.vv == '1e-3':
    #     Symbols_var[Symbols_var < 1e-3] = 1e-3  # < 0.001
    # else:
    #     print('wrong \n\n\n\n\n\nwrong \n\n\n\n\nwrong')
    # s_bar = (Symbols_mean[np.newaxis, :] @ K).reshape(-1)
    s_bar = np.sum(Symbols_mean)
    N = args.N
    obj, ff, pp, theta, cnt = optimizeRIS(args, M, N, h, Q, P, dd, Symbols_var2, args.LL, G, h_DP, 0)
    pp = pp * np.sqrt(np.power(10, cnt))

    # 仅用来查看：Pm2 * Vm2 <= P0
    ppp = Symbols_var * pp[:, 0]
    ppp2 = np.abs(ppp) ** 2
    print("Pm2 * Vm2 = ", ppp2)
    pp_ex = np.tile(pp, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols_pp = TransmittedSymbols * pp_ex
    SigPower = np.var(TransmittedSymbols_pp, axis=1)  # 四个发射功率
    SigPower_mean = np.mean(SigPower)  # 9.742
    print('SigPower_mean =', SigPower_mean)
    print(f'idx_packet = {idx_packet} \n')

    if SigPower_mean < 1e-4:
        SigPower_mean = 1e-4

    # D_entry = h.T.conj() @ ff * pp
    D_entry = np.zeros((M, 1), dtype=complex)
    for i in range(M):
        D_entry[i] = ff.T.conj() @ (h_DP[:, i, np.newaxis] + G[:, :, i] @ theta) * pp[i]

    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]

    SignalPart = np.sum(TransmittedSymbols, 0)
    Receive_SigPower_Per_Symbol = np.sum(np.power(np.abs(SignalPart), 2)) / L  # 186.119
    EsN0 = np.power(10, args.EsN0dB / 10.0)
    Receive_SigPower = np.sum(np.power(np.abs(TransmittedSymbols), 2), axis=1) / L  # 每个用户的接收功率
    N0 = SigPower_mean / EsN0
    # N0 = Receive_SigPower_Per_Symbol / EsN0
    # N0 = args.P0 / EsN0
    noisePowerVec = N0 / dd
    # SNR = 10 * np.log(args.P0 / noisePowerVec)

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)  # 相当于每一个元素重复4回: ndarray(4, 1040)
    for idx in np.arange(M):  # 见 pad “015”
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)  # (1043,)

    # generate noise
    for n in range(N):
        for idx in np.arange(M):
            noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
                loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
            if idx == 0:
                AWGNnoise = np.array([noise])
            else:
                AWGNnoise = np.r_[AWGNnoise, np.array([noise])]  # ndarray(4, 261)每一行表示一个匹配滤波对于整个数据包的260+1个噪声。
        AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')  # (1, 1044)
        if n == 0:
            AWGNnoise_f_reshape = np.array([AWGNnoise_reshape.flatten()])
        else:
            AWGNnoise_f_reshape = np.r_[AWGNnoise_f_reshape, np.array([AWGNnoise_reshape.flatten()])]
    noise_plus_f = ff.conj().T @ AWGNnoise_f_reshape
    samples = samples + noise_plus_f[0][0:-1]
    # samples = samples + AWGNnoise_reshape[0][0:-1]  # (1043, )

    D = np.zeros((M * (L + 1) - 1, M * L), dtype=complex)  # (1043, 1040)
    for i in range(M * L):
        D[np.arange(M) + i, i] = D_entry[np.mod(i, M)]
    DzVec = np.tile(noisePowerVec, [1, L + 1])  # 按维度1复制1次（不变），按维度2复制261次，得：(1, 1044)
    Dz = np.linalg.norm(ff, 2) ** 2 * np.diag(DzVec[0][:-1])  # (1043, 1043)

    # ------------------------------------- ML
    MUD = np.matmul(D.conj().T, np.linalg.inv(Dz))  # 矩阵求逆
    MUD = np.matmul(MUD, D)  # 矩阵相乘
    MUD = np.matmul(np.linalg.inv(MUD), D.conj().T)
    MUD = np.matmul(MUD, np.linalg.inv(Dz))
    MUD = np.matmul(MUD, np.array([samples]).T)  # np.array([samples])维度为：(1, 1043)；MUD：(1040, 1)

    # ------------------------------------- Estimate SUM
    output = (V @ MUD).reshape(-1)
    # output = np.sum(np.reshape(MUD, [L, M]), 1)  # (260,)
    output = output + s_bar
    output = output * args.lr
    return output


def per_pkt_transmission_LMMSE(args, L, TransmittedSymbols, V, idxs_users):
    M = max(int(args.frac * args.M_Prime), 1)
    taus = np.sort(np.random.uniform(0, args.maxDelay, (1, M - 1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(M)
    for idx in np.arange(M):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == M - 1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx - 1]
    dd[dd < 1e-4] = 1e-4
    h = 1 / np.sqrt(2) * (np.random.randn(args.N, M) + np.random.randn(args.N, M) * 1j)
    K = args.K[idxs_users]
    K = K / sum(K)
    K = K[:, np.newaxis]
    K_ex = np.tile(K, (1, len(TransmittedSymbols[0])))
    TransmittedSymbols = TransmittedSymbols * K_ex

    Symbols_mean = np.mean(TransmittedSymbols, axis=1)  # (4,)
    Symbols_mean_ex = np.tile(Symbols_mean[:, np.newaxis], [1, len(TransmittedSymbols[0])])
    Symbols_mean_con = np.tile(Symbols_mean, [1, L]).T

    # 是否零化均值
    isMean = True
    if isMean:
        # 减去mean
        TransmittedSymbols = TransmittedSymbols - Symbols_mean_ex
        # 此时均值为零，计算方差，即为功率
        Symbols_mean2 = np.mean(TransmittedSymbols, axis=1)  # (4,)
        Symbols_mean_con = np.tile(Symbols_mean2, [1, L]).T
        s_bar = np.sum(Symbols_mean)
    else:
        s_bar = 0

    Symbols_var = np.var(TransmittedSymbols, axis=1)  # (4,)
    TransmittedSymbols_pp = TransmittedSymbols
    SigPower = np.var(TransmittedSymbols_pp, axis=1)
    SigPower2 = np.sum(np.power(np.abs(TransmittedSymbols_pp), 2), axis=1) / L  # 均值零化之后，则与SigPower相等
    SigPower_mean = np.mean(SigPower)  # 0.00056

    # 完全对齐的情况
    D_entry = np.ones(M, dtype=complex)
    D_entry = D_entry[:, np.newaxis]
    for idx in range(M):
        TransmittedSymbols[idx, :] = TransmittedSymbols[idx, :] * D_entry[idx][0]

    # Symbols_var_con = Symbols_var[:, np.newaxis]
    # for i in range(L - 1):
    #     Symbols_var_con = np.vstack((Symbols_var_con, Symbols_var[:, np.newaxis]))
    Symbols_var_con = np.tile(Symbols_var, [1, L]).T
    D_tiled = np.diag(Symbols_var_con.reshape(-1))

    EsN0 = np.power(10, args.EsN0dB / 10.0)
    N0 = SigPower_mean / EsN0
    noisePowerVec = N0 / dd

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, M, axis=1)  # 相当于每一个元素重复4回: ndarray(4, 1040)
    for idx in np.arange(M):  # 见 pad “015”
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(M - idx - 1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    samples = np.sum(samples, axis=0)  # (1043,)

    # generate noise
    for idx in np.arange(M):
        noise = np.random.normal(loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1) + 1j * np.random.normal(
            loc=0, scale=np.sqrt(N0 / 2 / dd[idx]), size=L + 1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]  # ndarray(4, 261)

    AWGNnoise_reshape = np.reshape(AWGNnoise, (1, M * (L + 1)), 'F')  # (1, 1044)
    samples = samples + AWGNnoise_reshape[0][0:-1]  # (1043, )

    D = np.zeros((M * (L + 1) - 1, M * L), dtype=complex)  # (1043, 1040)
    for i in range(M * L):
        D[np.arange(M) + i, i] = D_entry[np.mod(i, M)]
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = np.diag(DzVec[0][:-1])  # (1043, 1043)

    # ------------------------------------- get A
    A = np.matmul(V, D_tiled)
    A = np.matmul(A, D.conj().T)
    B = np.matmul(D, D_tiled)
    B = np.matmul(B, D.conj().T)
    A = np.matmul(A, getInv(B + Dz))

    # ------------------------------------- Estimate SUM
    output = ((A @ np.array([samples]).T) + (V - A @ D) @ Symbols_mean_con).reshape(-1)
    if isMean:
        output = output + s_bar
    output = output * args.lr
    # output = output / len(K)
    return output
