"""
作者：hp
日期：2022年08月26日
"""
import copy

import cvxpy as cp
import numpy as np


def optimize(args, M, N, h, Q, P, dd, Vm2, cnt):
    f_norm2 = args.f_norm2
    P0 = args.P0
    L = args.L
    # print('Vm2=', Vm2)
    # print('h=', h)
    # print('D_A=', D_A)
    while np.max(Vm2) > 10:
        Vm2 = Vm2 * 1e-1
    while np.max(Vm2) < 1e-5:
        Vm2 = Vm2 * 1e1
        cnt += 1
    noisePowerVec = 1 / dd
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = args.f_norm2 * np.diag(DzVec[0][:-1])
    W = P.T.conj() @ np.linalg.inv(Dz) @ P
    cc = np.linalg.inv(np.linalg.cholesky(W))
    W_inv = np.dot(cc.T.conj(), cc)
    # W_inv2 = np.linalg.inv(W)
    A = W_inv * Q.T.conj()
    D_A = np.zeros((M, M))
    for i in range(0, M * L, M):
        for j in range(0, M * L, M):
            D_A += A[i:i + M, j:j + M]
    # D_A_eig = np.linalg.eig(D_A)[0]
    while np.max(np.linalg.eig(D_A)[0]) > 1e4:
        D_A = D_A * 1e-1

    Vm2 = Vm2[:, np.newaxis]
    P0_Vec = P0 / Vm2
    # P0_Vec_ex = np.concatenate((P0_Vec, P0_Vec), axis=0)

    c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
    d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
    for i in range(M):
        if np.abs(c_ran[0, i]) < 1e-3:
            c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
        if np.abs(d_ran[0, i]) < 1e-3:
            d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3

    xx = cp.Variable((M, 1), complex=True)
    ff = cp.Variable((N, 1), complex=True)

    for idx in range(args.SCA_I_max):
        obj = cp.Minimize(cp.quad_form(xx, D_A))
        constraints = [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ h[:, i]), cp.imag(cp.conj(ff.T) @ h[:, i])] - d[:, i])
            >= cp.inv_pos(P0_Vec[i, 0]) for i in range(M)]

        constraints += [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ h[:, i]), cp.imag(cp.conj(ff.T) @ h[:, i])] - d[:, i])
            <= cp.max(Vm2) for i in range(M)]

        constraints += [cp.norm(ff, 2) ** 2 <= f_norm2]

        prob = cp.Problem(obj, constraints)
        # prob.solve(solver=cp.SCS, verbose=False)
        # prob.solve(solver=cp.MOSEK, verbose=False)
        # prob.solve(solver=cp.CVXOPT, verbose=False)
        # prob.solve(verbose=True)
        try:
            # if args.test > 0:
            #     args.test -= 1
            #     raise cvxpy.error.SolverError("Solver xxx failed. ")
            prob.solve(verbose=False)
        except (cp.SolverError, Exception):
            print("Solver 'MOSEK' failed.")
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            ff.value = np.random.randn(N, 1) + np.random.randn(N, 1) * 1j
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            continue
            # break

        if prob.status == 'infeasible':
            print('problem became infeasible at iteration{}'.format(idx + 1))
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            ff.value = np.random.randn(N, 1) + np.random.randn(N, 1) * 1j
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            continue
            # break

        gap = 0
        for i in range(M):
            gap_c = np.linalg.norm(np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)]) - c[:, i], 2)
            gap_d = np.linalg.norm(
                np.array(
                    [float(np.real(ff.value.T.conj() @ h[:, i])), float(np.imag(ff.value.T.conj() @ h[:, i]))]) - d[:,
                                                                                                                  i], 2)
            gap = gap_d + gap_c
            c[:, i] = np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)])
            d[:, i] = [float(np.real(ff.value.T.conj() @ h[:, i])), float(np.imag(ff.value.T.conj() @ h[:, i]))]

        # print(f'gap = {gap}')
        # print(f'obj = {prob.value} \n')
        if gap <= args.epsilon:
            break

    result_obj = prob.value
    result_xx = xx.value
    result_ff = ff.value
    # print(f'result_obj = {result_obj}\n')
    # print(f'result_xx = {result_xx}\n')
    # print(f'result_ff = {result_ff}\n')
    # print(f'ff_norm2 = {np.linalg.norm(result_ff, 2) ** 2}\n')
    pp = ((1 / result_xx).T.conj() / (result_ff.T.conj() @ h)).T.conj()
    pp2 = np.abs(pp) ** 2
    # print(f'pp = {pp}\n')
    # print(f'pp2 = {pp2}\n')
    # print(f'P0_Vec = {P0_Vec}')
    obj_SCA = result_obj / L

    print(f'Avg_obj_SCA = {obj_SCA}')
    true_P_SCA = Vm2 * pp2
    # print(f'true_P_SCA = \n {true_P_SCA}')

    a = 1 / min(pp2)
    b = max(Vm2)
    if abs(b - a) <= 1e-7 or a <= b:
        print('True')
        args.count = 0
    else:
        print('False')
        args.count = args.count + 1
        if args.cur == 1:
            Vm2 = Vm2.reshape(-1)
            if args.count >= 3:
                result_obj, result_ff, pp, cnt = optimize(args, M, N, h, Q, P, dd, Vm2 * 10, cnt + 1)
            else:
                result_obj, result_ff, pp, cnt = optimize(args, M, N, h, Q, P, dd, Vm2, cnt)

    return result_obj, result_ff, pp, cnt  # 可以把round写进去


def optimizeRIS(args, M, N, h, Q, P, dd, Vm2, LL, G, h_DP, cnt):
    f_norm2 = args.f_norm2
    L = args.L
    while np.max(Vm2) > 10:
        Vm2 = Vm2 * 1e-1
    while np.max(Vm2) < 1e-5:
        Vm2 = Vm2 * 1e1
        cnt += 1
    noisePowerVec = 1 / dd
    DzVec = np.tile(noisePowerVec, [1, L + 1])
    Dz = args.f_norm2 * np.diag(DzVec[0][:-1])
    W = P.T.conj() @ np.linalg.inv(Dz) @ P
    cc = np.linalg.inv(np.linalg.cholesky(W))
    W_inv = np.dot(cc.T.conj(), cc)
    # W_inv2 = np.linalg.inv(W)
    A = W_inv * Q.T.conj()
    D_A = np.zeros((M, M))
    for i in range(0, M * L, M):
        for j in range(0, M * L, M):
            D_A += A[i:i + M, j:j + M]
    while np.max(np.linalg.eig(D_A)[0]) > 1e4:
        D_A = D_A * 1e-1

    try:
        xx_f, ff, obj_ori = for_ff(args, D_A, copy.deepcopy(Vm2), M, N, h, f_norm2)
    except (cp.SolverError, Exception):
        print("[optimizeRIS] for_ff failed, 初始化ff失败")
        return optimizeRIS(args, M, N, h, Q, P, dd, Vm2, LL, G, h_DP, cnt)

    last = 10
    i = 0
    for i in range(args.RIS_I_max):
        xx_theta, theta_V, obj_thetaV = for_theta_V02(args, D_A, copy.deepcopy(Vm2), M, LL, h, ff, G)
        for j in range(M):
            h[:, j] = (h_DP[:, j, np.newaxis] + G[:, :, j] @ theta_V).reshape(-1)
        try:
            xx_f, ff, obj_ff = for_ff(args, D_A, copy.deepcopy(Vm2), M, N, h, f_norm2)
        except (cp.SolverError, Exception):
            print(f"[optimizeRIS] for_ff failed, idx={i}")
            return optimizeRIS(args, M, N, h, Q, P, dd, Vm2, LL, G, h_DP, cnt)

        try:
            obj_sum = (obj_thetaV + obj_ff) / 2.
        except (cp.SolverError, Exception):
            print("[optimizeRIS] obj_sum = (obj_thetaV + obj_ff) / 2. failed")
            return optimizeRIS(args, M, N, h, Q, P, dd, Vm2, LL, G, h_DP, cnt)

        gap = abs(last - obj_sum)
        if gap < args.epsilon_RIS:
            print(f"[optimizeRIS] gap < {args.epsilon_RIS} gap足够小，提前结束, obj_sum={obj_sum}, gap={gap}")
            break
        if obj_sum < obj_ori * 0.1:
            print(f"[optimizeRIS] {obj_sum} < {obj_ori} * 0.1 目标值足够小，提前结束, obj_sum={obj_sum}, gap={gap}")
            break
        last = obj_sum
    if i == args.RIS_I_max - 1:
        print("[optimizeRIS] 达到最大迭代次数")
    print("[optimizeRIS] round: ", i)

    res_xx = xx_f
    res_ff = ff
    # 查看ff
    # ff_norm = np.linalg.norm(ff) ** 2
    res_theta = theta_V
    # 查看 theta_V
    # theta_V_abs = abs(theta_V)
    res_obj = obj_sum

    pp = np.zeros((M, 1), dtype=complex)
    for i in range(M):
        pp[i] = 1 / (res_ff.T.conj() @ (h_DP[:, i, np.newaxis] + G[:, :, i] @ res_theta) @ res_xx[i])
    pp2 = np.abs(pp) ** 2
    true_P_SCA = Vm2 * pp2
    a = 1 / min(pp2)
    b = max(Vm2)
    if abs(b - a) <= 1e-4 or a <= b:
        # print('[optimizeRIS] True')
        args.count = 0
        obj_ris = float(np.real(xx_f.T.conj() @ D_A @ xx_f))
        if obj_ori > obj_ris:
            print(f"[optimizeRIS] Okay 优化成功，obj_ori={obj_ori} VS obj_ris={obj_ris}\n")
        else:
            print(f"[optimizeRIS] No 优化失败，obj_ori={obj_ori} VS obj_ris={obj_ris}\n")
    else:
        print(f'[optimizeRIS] False，优化结果不满足某个约束，a={a}, b={b}')
        args.count = args.count + 1
        if args.cur == 1:
            Vm2 = Vm2.reshape(-1)
            if args.count >= 3:
                res_obj, res_ff, pp, res_theta, cnt = optimizeRIS(args, M, N, copy.deepcopy(h_DP), Q, P, dd, Vm2 * 10,
                                                                  LL,
                                                                  G, h_DP, cnt + 1)
            else:
                res_obj, res_ff, pp, res_theta, cnt = optimizeRIS(args, M, N, copy.deepcopy(h_DP), Q, P, dd, Vm2, LL, G,
                                                                  h_DP, cnt)
    # obj_SCA = res_obj / L
    # print(f'[optimizeRIS] Avg_obj_SCA = {obj_SCA}')
    return res_obj, res_ff, pp, res_theta, cnt  # 可以把round写进去


def for_ff(args, D_A, Vm2, M, N, h, f_norm2):
    c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    for i in range(M):
        if np.abs(c_ran[0, i]) < 1e-3:
            c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
        if np.abs(d_ran[0, i]) < 1e-3:
            d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
    c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
    d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)

    xx = cp.Variable((M, 1), complex=True)
    ff = cp.Variable((N, 1), complex=True)

    Vm2 = Vm2[:, np.newaxis]
    P0_Vec = args.P0 / Vm2

    gap2 = 0
    idx = 0
    for idx in range(args.SCA_I_max):
        obj = cp.Minimize(cp.quad_form(xx, D_A))
        constraints = [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ h[:, i]), cp.imag(cp.conj(ff.T) @ h[:, i])] - d[:, i])
            >= cp.inv_pos(P0_Vec[i, 0]) for i in range(M)]

        constraints += [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ h[:, i]), cp.imag(cp.conj(ff.T) @ h[:, i])] - d[:, i])
            <= cp.max(Vm2) for i in range(M)]

        constraints += [cp.norm(ff, 2) ** 2 <= f_norm2]

        prob = cp.Problem(obj, constraints)
        # prob.solve(solver=cp.SCS, verbose=False)
        # prob.solve(solver=cp.MOSEK, verbose=False)
        # prob.solve(verbose=True)
        try:
            # if args.test > 0:
            #     args.test -= 1
            #     raise cvxpy.error.SolverError("Solver xxx failed. ")
            prob.solve(verbose=False)
        # except cp.SolverError:
        # except cp.SolverError or Exception:
        except (cp.SolverError, Exception):
            print(c_ran)
            print(d_ran)
            print(D_A)
            print(Vm2)
            print(d_ran)
            print(xx)
            print(ff)
            print("[for_ff] Solver 'MOSEK' failed.")
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            ff.value = np.random.randn(N, 1) + np.random.randn(N, 1) * 1j
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            continue

        if prob.status == 'infeasible':
            print('[for_ff] problem became infeasible at iteration{}'.format(idx + 1))
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            ff.value = np.random.randn(N, 1) + np.random.randn(N, 1) * 1j
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            continue

        gap = 0
        for i in range(M):
            gap_c = np.linalg.norm(np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)]) - c[:, i], 2)
            gap_d = np.linalg.norm(
                np.array(
                    [float(np.real(ff.value.T.conj() @ h[:, i])), float(np.imag(ff.value.T.conj() @ h[:, i]))]) - d[:,
                                                                                                                  i], 2)
            gap = gap_d + gap_c
            c[:, i] = np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)])
            d[:, i] = [float(np.real(ff.value.T.conj() @ h[:, i])), float(np.imag(ff.value.T.conj() @ h[:, i]))]

        # print(f"[for_ff] test: gap={gap}")
        if gap <= args.epsilon_RIS_forff:
            print(f"[for_ff] gap <= {args.epsilon_RIS_forff}：gap足够小，提前返回, gap={gap}")
            break
        if abs(gap - gap2) <= args.gap_gap_forff:
            print(f"[for_ff] abs(gap - gap2) <= {args.gap_gap_forff}：gap没变化，提前返回, gap={gap}, gap2={gap2}")
            break
        gap2 = gap
    if idx == args.SCA_I_max - 1:
        print("[for_ff] 达到最大迭代次数")
    print(f"[for_ff] F_idx={idx}")
    ff = ff.value
    xx = xx.value
    return xx, ff, prob.value


def for_theta_V02(args, D_A, Vm2, M, LL, h, ff, G):
    obj_real_CCP = 1.
    # 这里可以改成matlab中的随机初始化方式：
    # ram_x = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
    # ram_theta = np.random.randn(LL, 1) + np.random.randn(LL, 1) * 1j
    c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
    for i in range(M):
        if np.abs(c_ran[0, i]) < 1e-3:
            c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
        if np.abs(d_ran[0, i]) < 1e-3:
            d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
    c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
    d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)

    Vm2 = Vm2[:, np.newaxis]
    P0_Vec = args.P0 / Vm2

    xx = cp.Variable((M, 1), complex=True)
    theta_V = cp.Variable((LL, 1), complex=True)
    a1 = cp.Variable((2 * LL, 1), nonneg=True)

    gap2 = 0
    rho = 0.6
    theta_last = np.exp(1j * 2 * np.pi * np.random.rand(LL, 1))

    idx = 0
    for idx in range(args.SCA_I_max):
        # obj = cp.Minimize(cp.quad_form(xx, D_A) + rho * sum(a1) + rho * sum(b1))
        obj = cp.Minimize(cp.quad_form(xx, D_A) + rho * sum(a1))

        constraints = [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V)),
                     cp.imag(cp.conj(ff.T) @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V))] - d[:, i])
            >= cp.inv_pos(P0_Vec[i, 0]) for i in range(M)]

        constraints += [
            np.linalg.norm(c[:, i], 2) ** 2 * np.linalg.norm(d[:, i], 2) ** 2 +
            2 * np.linalg.norm(d[:, i], 2) ** 2 * c[:, i].T @ ([cp.real(xx[i, 0]), cp.imag(xx[i, 0])] - c[:, i]) +
            2 * np.linalg.norm(c[:, i], 2) ** 2 * d[:, i].T @ (
                    [cp.real(cp.conj(ff.T) @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V)),
                     cp.imag(cp.conj(ff.T) @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V))] - d[:,
                                                                                              i])
            <= cp.max(Vm2) for i in range(M)]

        constraints += [cp.square(cp.abs(theta_V[i])) <= a1[i] + 1 for i in range(LL)]
        constraints += [
            cp.real(cp.abs(theta_last[i]) ** 2 - 2 * (cp.conj(theta_last[i, 0]) * theta_V[i])) <= a1[i + LL] - 1
            for i in range(LL)]

        prob = cp.Problem(obj, constraints)
        # prob.solve(solver=cp.SCS, verbose=False)
        # prob.solve(solver=cp.MOSEK, verbose=False)
        # prob.solve(verbose=True)
        try:
            prob.solve(verbose=False)
        except (cp.SolverError, Exception):
            print("[for_theta_V02] Solver 'MOSEK' failed.")
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            theta_V.value = np.random.randn(LL, 1) + np.random.randn(LL, 1) * 1j
            a1.value = abs(np.random.randn(2 * LL, 1))
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            continue

        if prob.status == 'infeasible':
            print('[for_theta_V02] problem became infeasible at iteration{}'.format(idx + 1))
            xx.value = np.random.randn(M, 1) + np.random.randn(M, 1) * 1j
            theta_V.value = np.random.randn(LL, 1) + np.random.randn(LL, 1) * 1j
            a1.value = abs(np.random.randn(2 * LL, 1))
            c_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            d_ran = np.random.randn(1, M) + np.random.randn(1, M) * 1j
            for i in range(M):
                if np.abs(c_ran[0, i]) < 1e-3:
                    c_ran[0, i] = c_ran[0, i] / np.abs(c_ran[0, i]) * 1e-3
                if np.abs(d_ran[0, i]) < 1e-3:
                    d_ran[0, i] = d_ran[0, i] / np.abs(d_ran[0, i]) * 1e-3
            c = np.concatenate((np.real(c_ran), np.imag(c_ran)), axis=0)
            d = np.concatenate((np.real(d_ran), np.imag(d_ran)), axis=0)
            continue

        theta_last = theta_V.value
        obj_real_CCP = np.real(xx.value.T.conj() @ D_A @ xx.value)
        gap = 0
        for i in range(M):
            gap_c = np.linalg.norm(np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)]) - c[:, i], 2)
            gap_d = np.linalg.norm(
                np.array(
                    [float(np.real(ff.T.conj() @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V.value))),
                     float(np.imag(ff.T.conj() @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V.value)))]) - d[:,
                                                                                                          i], 2)
            gap = gap_d + gap_c
            c[:, i] = np.array([np.real(xx[i, 0].value), np.imag(xx[i, 0].value)])
            d[:, i] = [float(np.real(ff.T.conj() @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V.value))),
                       float(np.imag(ff.T.conj() @ (h[:, i, np.newaxis] + G[:, :, i] @ theta_V.value)))]

        # print(f"[for_theta_V02] test: gap={gap}")
        if gap <= args.epsilon_RIS_forTheta:
            print(f"[for_theta_V02] gap <= {args.epsilon_RIS_forTheta}：gap足够小，提前返回, gap={gap}")
            break
        if abs(gap - gap2) <= args.gap_gap_forTheta:
            print(f"[for_theta_V02] abs(gap - gap2) <= {args.gap_gap_forTheta}：gap没变化，提前返回, gap={gap}, gap2={gap2}")
            break
        gap2 = gap
        # print("obj_real_CCP = ", float(obj_real_CCP))
    if idx == args.SCA_I_max - 1:
        print("[for_ff] 达到最大迭代次数")
    print("[for_theta_V02] RIS_idx=", idx)
    # print("obj_real_CCP = ", float(obj_real_CCP))
    # theta_V_abs = abs(theta_V.value)
    # print("theta_V_abs = ", theta_V_abs)

    return xx.value, theta_V.value, float(obj_real_CCP)
