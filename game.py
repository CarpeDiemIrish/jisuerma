import numpy as np
import random
import copy
import threading
import math
import time
import pymysql
import tensorflow as tf
from multiprocessing import Pool, Manager  # Process,
import os


# {'th': 0.85, 'dh': 1.42, 'xj_tt': 3.03, 'zj_tt': 6.91, 'bt': 66.25, 'p': 7.41, 'mg': 4.35, 'ag': 2.45,
# 'gskh': 2.7, 'bhs': 851241, 'dp': 8.81, 'zm': 51.27, 'yj': 77.54, 'zj': 49.98, 'fs': 62386333}
wjxx = {'fs': [1, 1], 'model_no': [0, 0], 'model_cp': [None, None], 'model_cp_wc': [0, 0.05], 'model_bt': [None, None], 'model_p': [None, None],
        'bt': [[0.45, 0.45, 0.5], [0.45, 0.45, 0.5]],
        'p': [0.38, 0.38],
        'zjpj': [[], []], 'zkpj': [None, None], 'qbpj': [[], []]}
if 1 in wjxx['fs']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'  # 运行程序，都会占用gpu0全部资源
    for i in range(2):
        if wjxx['fs'][i] == 1:
            cwd_h5 = os.getcwd() + '/model_'+str(wjxx['model_no'][i])+'/'
            model_path_full = cwd_h5 + 'cp.h5'
            wjxx['model_cp'][i] = tf.keras.models.load_model(model_path_full)
            model_path_full = cwd_h5 + 'bt.h5'
            wjxx['model_bt'][i] = tf.keras.models.load_model(model_path_full)
            model_path_full = cwd_h5 + 'p.h5'
            wjxx['model_p'][i] = tf.keras.models.load_model(model_path_full)


def sysl_del_pmsz(sysl, pmsz):
    for pm in pmsz:
        sysl[pm-1] -= 1


def pmsz_del(pmsz, delsz):
    for pm_ in delsz:
        if pm_ in pmsz:
            pmsz.remove(pm_)


def getnp_pmsl(np_a, pmsl):
    np_b = np.zeros((4, 12), dtype=np.int8)
    for i in range(12):
        n = pmsl[i]
        for i2 in range(0, n):
            np_b[i2][i] = 1
    np_a = np.append(np_a, np_b)
    return np_a


def getnp_pmsz(np_a, pmsz):
    np_b = np.zeros((4, 12), dtype=np.int8)
    listpm = [pm-1 for pm in pmsz]
    for i in range(12):
        n = listpm.count(i)
        for i2 in range(0, n):
            np_b[i2][i] = 1
    np_a = np.append(np_a, np_b)
    return np_a


def getnp_pm(np_a, pm):
    np_b = np.zeros((4, 12), dtype=np.int8)
    if pm > 0:
        np_b[0][pm-1] = 1
    np_a = np.append(np_a, np_b)
    return np_a


def get_np_pjxx(pjxx, wz_wf):
    np_a = np.array([], dtype=np.int8)
    np_b = np.zeros((4, 12), dtype=np.int8)
    # 庄家
    if pjxx['zj'] == wz_wf:
        np_b[0][0] = 1
    # 番牌
    np_b[1][pjxx['fp']-1] = 1
    # 对方状态(1/2)
    pm_ = pjxx['wjxx'][1-wz_wf]['btzt']
    if pm_ > 0:
        np_b[0][pm_] = 1
    # 对面明牌区(碰)
    pm_ = pjxx['wjxx'][1-wz_wf]['mpq_p']
    if pm_ > 0:
        np_b[2][pm_-1] = 1
    # 对面明牌区(明杠)
    pm_ = pjxx['wjxx'][1-wz_wf]['mpq_mg']
    if pm_ > 0:
        np_b[3][pm_-1] = 1
    # 对面明牌区(暗杠)
    if pjxx['wjxx'][1-wz_wf]['mpq_ag'] > 0:
        np_b[0][3] = 1
    np_a = np.append(np_a, np_b)
    # 对方已出牌(听前)
    np_a = getnp_pmsz(np_a, pjxx['wjxx'][1-wz_wf]['ycp_tq'])
    # 对方已出牌(听后)
    np_a = getnp_pmsz(np_a, pjxx['wjxx'][1-wz_wf]['ycp_th'])
    # 自己牌面
    np_a = getnp_pmsz(np_a, pjxx['wjxx'][wz_wf]['pm'])
    np_b = np.zeros((4, 12), dtype=np.int8)
    # 自己明牌区(碰)
    pm_ = pjxx['wjxx'][wz_wf]['mpq_p']
    if pm_ > 0:
        np_b[0][pm_-1] = 1
    # 对面明牌区(明杠)
    pm_ = pjxx['wjxx'][wz_wf]['mpq_mg']
    if pm_ > 0:
        np_b[1][pm_-1] = 1
    # 对面明牌区(暗杠)
    pm_ = pjxx['wjxx'][wz_wf]['mpq_ag']
    if pm_ > 0:
        np_b[2][pm_-1] = 1
    # 剩余牌数
    if pjxx['fbh_syps'] >= 30:
        np_b[3][2] = 1
    elif pjxx['fbh_syps'] >= 20:
        np_b[3][1] = 1
    elif pjxx['fbh_syps'] >= 10:
        np_b[3][0] = 1
    pm_ = pjxx['fbh_syps'] % 10
    if pm_ > 0:
        np_b[3][pm_+2] = 1
    # 自已已出牌(对方听前)
    np_a = getnp_pmsz(np_a, pjxx['wjxx'][wz_wf]['ycp_tq_df'])
    # 自已已出牌(对方听后)
    np_a = getnp_pmsz(np_a, pjxx['wjxx'][wz_wf]['ycp_th_df'])
    # 剩余牌面
    np_a = getnp_pmsl(np_a, pjxx['wjxx'][wz_wf]['sysl'][:12])
    np_a = np.append(np_a, np_b)
    return np_a


def getnp_cp(pjxx, wz_wf):
    np_a = get_np_pjxx(pjxx, wz_wf)
    pmsl_ = [0]*12
    pmsl_[0] = pjxx['wjxx'][wz_wf]['bhs']
    pmsl_[1] = pjxx['wjxx'][1-wz_wf]['bhs']
    pmsl_[2] = pjxx['wjxx'][wz_wf]['sysl'][12]
    np_a = getnp_pmsl(np_a, pmsl_)
    np_a = np.reshape(np_a, (1, 9, 4, 12))
    np_a = np_a.astype(np.float16)
    # print(np_a)
    predictions = wjxx['model_cp'][wz_wf].predict(np_a, verbose=0)
    pre_np = np.array(predictions[0])
    b = -np.sort(-pre_np)
    c = np.argsort(-pre_np)
    del_np = []
    for i in range(len(c)):
        if c[i]+1 not in pjxx['wjxx'][wz_wf]['pm']:
            del_np.append(i)
    b = np.delete(b, del_np)
    c = np.delete(c, del_np)
    cp_pm_sz = [c[0]+1]
    # print(predictions[0])
    if wjxx['model_cp_wc'][wz_wf] != 0:
        for i in range(len(b)-1):
            if b[0] - b[i+1] <= wjxx['model_cp_wc'][wz_wf]:
                cp_pm_sz.append(c[i+1]+1)
    # print("最佳出牌", cp_pm_sz)
    return cp_pm_sz


def getnp_bt(pjxx, wz_wf, cpm, btlx):
    np_a = get_np_pjxx(pjxx, wz_wf)
    np_b = np.zeros((4, 12), dtype=np.int8)
    if cpm != 0:
        np_b[0][cpm-1] = 1
    np_a = np.append(np_a, np_b)
    pmsl_ = [0]*12
    pmsl_[0] = pjxx['wjxx'][wz_wf]['bhs']
    pmsl_[1] = pjxx['wjxx'][1-wz_wf]['bhs']
    pmsl_[2] = pjxx['wjxx'][wz_wf]['sysl'][12]
    pmsl_[2+btlx] = 4
    np_a = getnp_pmsl(np_a, pmsl_)
    np_a = np.reshape(np_a, (1, 10, 4, 12))
    np_a = np_a.astype(np.float16)
    # print(np_a)
    # predictions = model_bt.predict(np_a)
    predictions = wjxx['model_bt'][wz_wf].predict(np_a, verbose=0)
    # print(predictions[0])
    if predictions[0][0] > wjxx['bt'][wz_wf][btlx-1]:
        return True
    else:
        return False


def getnp_p(pjxx, wz_wf, dfcp):
    np_a = get_np_pjxx(pjxx, wz_wf)
    np_b = np.zeros((4, 12), dtype=np.int8)
    np_b[0][dfcp-1] = 1
    np_a = np.append(np_a, np_b)
    pmsl_ = [0]*12
    pmsl_[0] = pjxx['wjxx'][wz_wf]['bhs']
    pmsl_[1] = pjxx['wjxx'][1-wz_wf]['bhs']
    pmsl_[2] = pjxx['wjxx'][wz_wf]['sysl'][12]
    np_a = getnp_pmsl(np_a, pmsl_)
    np_a = np.reshape(np_a, (1, 10, 4, 12))
    np_a = np_a.astype(np.float16)
    predictions = wjxx['model_p'][wz_wf].predict(np_a, verbose=0)
    # print(predictions[0])
    if predictions[0][0] > wjxx['p'][wz_wf]:
        return True
    else:
        return False


def count_fs(pxxx):
    n_z = n_x = fs = 0
    if pxxx['dz'] >= 10:
        fs += 20
        n_z += 2
    else:
        fs += pxxx['dz'] * 2
        n_x += 2

    if pxxx['kz'] > 0:
        if pxxx['kz'] >= 10:
            fs += 30
            n_z += 3
        else:
            fs += pxxx['kz'] * 3
            n_x += 3

    if pxxx['sz'] > 0:
        fs += pxxx['sz'] * 3 + 3
        n_x += 3

    # 清一色
    if n_z == 0:
        fs += 10
    # 字一色
    if n_x == 0:
        fs += 30
    # 碰碰胡
    if pxxx['sz'] == 0:
        fs += 20
    return fs


def allpxxx():
    all_pxxx_ = []
    for i in range(12):
        pxxx_ = {'pmsz': [], 'pmsl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'dz': 0, 'sz': 0, 'kz': 0}
        pxxx_['pmsz'].append(i + 1)
        pxxx_['pmsz'].append(i + 1)
        pxxx_['pmsl'][i] = 2
        pxxx_['dz'] = i + 1
        # 顺子
        for j in range(7):
            pxxx_2 = copy.deepcopy(pxxx_)
            for h in range(3):
                pxxx_2['pmsz'].append(j + 1 + h)
                pxxx_2['pmsl'][j + h] += 1
            pxxx_2['sz'] = j + 1
            pxxx_2['fs'] = count_fs(pxxx_2)
            all_pxxx_.append(pxxx_2)
        # 刻字
        for j in range(12):
            if i == j:
                continue
            pxxx_2 = copy.deepcopy(pxxx_)
            for h in range(3):
                pxxx_2['pmsz'].append(j + 1)
            pxxx_2['pmsl'][j] += 3
            pxxx_2['kz'] = j + 1
            pxxx_2['fs'] = count_fs(pxxx_2)
            all_pxxx_.append(pxxx_2)
    return all_pxxx_


all_pxxx = allpxxx()


def pmsl_get_dcp(pmsl_, mpqpm_):
    dcp = []
    dcfs = []
    pmsl_2 = pmsl_[:12]
    pm_mpq = 0
    pm_ = 0
    if len(mpqpm_) > 0:
        pm_mpq = mpqpm_[0]
        pmsl_2[pm_mpq - 1] += 3
    for i in range(216):
        if pm_mpq > 0:
            if all_pxxx[i]['kz'] != pm_mpq:
                continue
        n_dff = 0
        for j in range(12):
            if pmsl_2[j] > all_pxxx[i]['pmsl'][j]:
                n_dff = 2
                break
            elif pmsl_2[j] < all_pxxx[i]['pmsl'][j]:
                n_dff += all_pxxx[i]['pmsl'][j] - pmsl_2[j]
                if n_dff > 1:
                    break
                pm_ = j + 1
        if n_dff == 1:
            dcp.append(pm_)
            dcfs.append(all_pxxx[i]['fs'])
    if len(dcp) > 0:
        return dcp, dcfs, True
    else:
        return dcp, dcfs, False


def pmsl_get_bss(pmsl_, mpqpm_):
    bss_ = 100
    pmsl_2 = pmsl_[:12]
    if len(mpqpm_) > 0:
        return 1
    for i in range(216):
        n_dff = 0
        for j in range(12):
            if pmsl_2[j] < all_pxxx[i]['pmsl'][j]:
                n_dff += all_pxxx[i]['pmsl'][j] - pmsl_2[j]
        if n_dff < bss_:
            bss_ = n_dff
    return bss_


def pmsl_is_hu(pmsl_, mpqpm_):
    pmsl_2 = pmsl_[:12]
    pm_mpq = 0
    if len(mpqpm_) > 0:
        pm_mpq = mpqpm_[0]
        pmsl_2[pm_mpq - 1] += 3
    for i in range(216):
        if pmsl_2 == all_pxxx[i]['pmsl']:
            if pm_mpq > 0:
                if all_pxxx[i]['kz'] != pm_mpq:
                    continue
            return all_pxxx[i]['fs']
    return 0


def score_cal(pjxx, wz, pmsl, extrascore):
    jfs = extrascore
    jfs += pmsl[pjxx['fp'] - 1] * 10
    if pjxx['wjxx'][wz]['mpq_mg'] > 0 or pjxx['wjxx'][wz]['mpq_ag'] > 0:
        # 杠 +50
        jfs += 50
        mpq_pm = pjxx['wjxx'][wz]['mpq_pmsz'][0]
        if mpq_pm == pjxx['fp']:
            jfs += 10
        if mpq_pm >= 10:
            jfs += 10
        else:
            jfs += mpq_pm
    if pjxx['wjxx'][wz]['btzt'] == 1:
        # 天听
        jfs += 20
    elif pjxx['wjxx'][wz]['btzt'] == 2:
        # 报听
        jfs += 10
    # jfs += pjxx['wjxx'][wz]['bhs'] * 20
    return jfs


def best_cp(pjxx, wz):
    qcpfz, bss = fzcl(pjxx, wz)
    px_pm = []
    px_fz = []
    px_bss = []
    for i in range(12):
        if not px_pm:
            px_pm.append(i + 1)
            px_fz.append(qcpfz[i])
            px_bss.append(bss[i])
        else:
            if i + 1 not in pjxx['wjxx'][wz]['pm']:
                px_pm.append(i + 1)
                px_fz.append(qcpfz[i])
                px_bss.append(bss[i])
                continue
            i_insert = 0
            for j in range(len(px_pm)):
                insert_ = 0
                if px_pm[j] not in pjxx['wjxx'][wz]['pm']:
                    insert_ = 1
                elif qcpfz[i] > px_fz[j]:
                    insert_ = 1
                if insert_ == 1:
                    px_pm.insert(j, i + 1)
                    px_fz.insert(j, qcpfz[i])
                    px_bss.insert(j, bss[i])
                    i_insert = 1
                    break
            if i_insert == 0:
                px_pm.append(i + 1)
                px_fz.append(qcpfz[i])
                px_bss.append(bss[i])
    # print(qcpfz)
    # print(px_pm, px_fz)
    return px_pm, px_bss


def pm_zfz(pjxx, wz):
    n_zpm = pjxx['fbh_syps']
    kqps = int(n_zpm / 2)
    if kqps <= 0:
        return 0
    if kqps > 3:
        kqps = 3
    zfz = 0
    zjpm_ = pjxx['wjxx'][wz]['pm'][:]
    pmsl_ = pjxx['wjxx'][wz]['pmsl'][:]
    n_ = len(pjxx['wjxx'][wz]['mpq_pmsz'])
    if n_ > 0:
        mpq_pm = pjxx['wjxx'][wz]['mpq_pmsz'][0]
        for i in range(3):
            zjpm_.append(mpq_pm)
            pmsl_[mpq_pm - 1] += 1
    else:
        mpq_pm = 0
    for i in range(len(all_pxxx)):
        if mpq_pm > 0:
            if all_pxxx[i]['kz'] != mpq_pm:
                continue
        cpxx_ = {'bps': 0, 'qp_pm': [], 'qp_ps': []}
        for j in range(12):
            if pmsl_[j] == all_pxxx[i]['pmsl'][j]:
                continue
            if pmsl_[j] < all_pxxx[i]['pmsl'][j]:
                cpxx_['bps'] += all_pxxx[i]['pmsl'][j] - pmsl_[j]
                cpxx_['qp_pm'].append(j + 1)
                cpxx_['qp_ps'].append(all_pxxx[i]['pmsl'][j] - pmsl_[j])
        if cpxx_['bps'] > 3:
            continue
        if pjxx['wjxx'][wz]['btzt'] != 0:
            if cpxx_['bps'] > 1:
                continue
        cpxx_['fs'] = all_pxxx[i]['fs']
        cpxx_['fs'] += score_cal(pjxx, wz, all_pxxx[i]['pmsl'], 0)
        fz = cpxx_calculate(pjxx, kqps, cpxx_)
        zfz += fz
    return zfz


def fzcl(pjxx, wz):
    n_zpm = pjxx['fbh_syps']
    kqps = int(n_zpm / 2)
    if kqps <= 0:
        return [0] * 12, [10] * 12
    if kqps > 3:
        kqps = 3
    # print(pjxx, wz)
    zjpm_ = pjxx['wjxx'][wz]['pm'][:]
    pmsl_ = pjxx['wjxx'][wz]['pmsl'][:]
    n_ = len(pjxx['wjxx'][wz]['mpq_pmsz'])
    if n_ > 0:
        mpq_pm = pjxx['wjxx'][wz]['mpq_pmsz'][0]
        for i in range(3):
            zjpm_.append(mpq_pm)
            pmsl_[mpq_pm - 1] += 1
    else:
        mpq_pm = 0
    # all_kcpx = []
    qcpfz = [0] * 12
    bss = [10] * 12
    for i in range(len(all_pxxx)):
        if mpq_pm > 0:
            if all_pxxx[i]['kz'] != mpq_pm:
                continue
        kcpx_ = {'bps': 0, 'wsypm': [], 'pxxx_n': i, 'cpxx': []}
        cpxx_ = {'qp_pm': [], 'qp_ps': []}
        for j in range(12):
            if pmsl_[j] == all_pxxx[i]['pmsl'][j]:
                continue
            if pmsl_[j] > all_pxxx[i]['pmsl'][j]:
                for h in range(pmsl_[j] - all_pxxx[i]['pmsl'][j]):
                    kcpx_['wsypm'].append(j + 1)
            else:
                kcpx_['bps'] += all_pxxx[i]['pmsl'][j] - pmsl_[j]
                cpxx_['qp_pm'].append(j + 1)
                cpxx_['qp_ps'].append(all_pxxx[i]['pmsl'][j] - pmsl_[j])
        if kcpx_['bps'] > 3:
            continue
        cpxx_['fs'] = all_pxxx[i]['fs']
        cpxx_['fs'] += score_cal(pjxx, wz, all_pxxx[i]['pmsl'], 0)
        kcpx_['cpxx'].append(cpxx_)
        # print(cpxx_,pjxx)
        kcpx_['fz'] = cpxx_calculate(pjxx, kqps, cpxx_)
        # print(kcpx_['fz'], kqps, cpxx_)
        for j in range(len(kcpx_['wsypm'])):
            if j != 0:
                if kcpx_['wsypm'][j] == kcpx_['wsypm'][j - 1]:
                    continue
            qcpfz[kcpx_['wsypm'][j] - 1] += kcpx_['fz']
            if kcpx_['bps'] < bss[kcpx_['wsypm'][j] - 1]:
                bss[kcpx_['wsypm'][j] - 1] = kcpx_['bps']
        # all_kcpx.append(kcpx_)
    return qcpfz, bss


def calculate_C(m, n):
    r = math.factorial(n) // (math.factorial(m) * math.factorial(n - m))
    return r


def cpxx_calculate(pjxx, kqps, cpxx):
    cppm_qpm = cpxx['qp_pm']
    cppm_qps = cpxx['qp_ps']
    cppm_syps = [pjxx['zpm'].count(pm) for pm in cppm_qpm]
    syps_z = 0
    # print(cppm_qpm,cppm_qps,cppm_syps)
    for i in range(len(cppm_syps)):
        syps_z += cppm_syps[i]
    dypm = []
    for i in range(len(cppm_qps)):
        if cppm_syps[i] < cppm_qps[i]:
            return 0
        else:
            dypm.append(cppm_syps[i] - cppm_qps[i] + 1)
    mjjg = dg_mjzh(dypm)
    n = len(dypm)
    zfz = 0
    zzhs = calculate_C(kqps, pjxx['fbh_syps'])
    for i in range(len(mjjg)):
        cs = 1
        ps = 0
        for j in range(n):
            cs *= calculate_C(mjjg[i][j] + cppm_qps[j] - 1, cppm_syps[j])
            ps += mjjg[i][j] + cppm_qps[j] - 1
        if ps > kqps:
            continue
        cz = kqps - ps
        # print(i,cz,cs,mjjg[i],cppm_qps,cppm_syps,ps)
        if cz > 0:
            cs *= calculate_C(cz, pjxx['fbh_syps'] - syps_z)
        zfz += cs / zzhs * cpxx['fs']
    return zfz


def dg_mjzh(source: list) -> list:
    n = len(source)
    if n == 1:
        return [[i] for i in range(1, source[0] + 1)]
    ans = []
    # 先把n拿出来放进组合，然后从生下的n-1个数里选出r-1个数
    if source[0] == 0:
        for each_list in dg_mjzh(source[1:]):
            ans.append([0] + each_list)
    else:
        for i in range(source[0]):
            for each_list in dg_mjzh(source[1:]):
                ans.append([i + 1] + each_list)
    if not ans:
        return [source]
    return ans


def pjxx_csh():
    pjxx = {
        'zpm': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
                7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13],
        'wjxx': [],
        'czgc': [],
        'loser_zfz': 0
    }
    random.shuffle(pjxx['zpm'])
    pjxx['fp'] = random.randint(1, 12)
    pjxx['zj'] = random.randint(0, 1)

    wjxx_c = {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0,
              'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0}

    wjxx_ = copy.deepcopy(wjxx_c)
    if pjxx['zj'] == 0:
        wjxx_['pm'] = pjxx['zpm'][0:9:2]
    else:
        wjxx_['pm'] = pjxx['zpm'][1:8:2]

    wjxx_['pmsl'] = [wjxx_['pm'].count(i + 1) for i in range(13)]
    wjxx_['sysl'] = [4-wjxx_['pmsl'][i] for i in range(13)]
    pjxx['wjxx'].append(wjxx_)

    wjxx_ = copy.deepcopy(wjxx_c)
    if pjxx['zj'] == 1:
        wjxx_['pm'] = pjxx['zpm'][0:9:2]
    else:
        wjxx_['pm'] = pjxx['zpm'][1:8:2]
    wjxx_['pmsl'] = [wjxx_['pm'].count(i + 1) for i in range(13)]
    wjxx_['sysl'] = [4 - wjxx_['pmsl'][i] for i in range(13)]
    pjxx['wjxx'].append(wjxx_)

    del pjxx['zpm'][0:9]
    buhua(pjxx, 0)
    buhua(pjxx, 1)
    pjxx['syps'] = len(pjxx['zpm'])
    pjxx['fbh_syps'] = 39
    return pjxx


def add_czgc(pjxx_, wz, cz, pm, bz):
    czgc = {'wz': wz, 'cz': cz, 'pm': pm, 'bz': bz, 'zjpm': pjxx_['wjxx'][wz]['pm'][:]}
    pjxx_['czgc'].append(czgc)
    # 1 出牌
    # 2 暗杠
    # 3 碰后杠
    # 4 明杠
    # 5 碰
    # 6 补牌
    # 7 补花
    # 8 闲家天听
    # 9 庄家天听
    # 10 报听


def buhua(pjxx_, wz):
    n_ = pjxx_['wjxx'][wz]['pmsl'][12]
    while n_ > 0:
        # 移除花牌
        for i in range(n_):
            pjxx_['wjxx'][wz]['pm'].remove(13)
            pjxx_['wjxx'][1-wz]['sysl'][12] -= 1
            add_czgc(pjxx_, wz, 7, 13, '补花')
        pjxx_['wjxx'][wz]['pmsl'][12] = 0
        pjxx_['wjxx'][wz]['bhs'] += n_
        # 补牌
        pjxx_['wjxx'][wz]['pm'].extend(pjxx_['zpm'][:n_])
        for pm_ in pjxx_['zpm'][:n_]:
            pjxx_['wjxx'][wz]['pmsl'][pm_ - 1] += 1
            pjxx_['wjxx'][wz]['sysl'][pm_ - 1] -= 1
        del pjxx_['zpm'][:n_]
        n_ = pjxx_['wjxx'][wz]['pmsl'][12]


def score_buhua_gang(pjxx, wz):
    fs = 0
    if pjxx['wjxx'][wz]['mpq_mg'] > 0 or pjxx['wjxx'][wz]['mpq_ag'] > 0:
        fs += 50
    if pjxx['wjxx'][1 - wz]['mpq_mg'] > 0 or pjxx['wjxx'][1 - wz]['mpq_ag'] > 0:
        fs -= 50
    fs += pjxx['wjxx'][wz]['bhs'] * 20
    fs -= pjxx['wjxx'][1 - wz]['bhs'] * 20
    return fs


def czgc_last(czgc, wz, last_cz, now_cz):
    l_czgc = len(czgc)
    if l_czgc < 2:
        return False
    if czgc[l_czgc-1]['cz'] != now_cz or czgc[l_czgc-1]['wz'] != wz:
        return False
    for i in range(l_czgc - 1):
        cz_ = czgc[l_czgc-i-2]['cz']
        if cz_ == 7:
            # 补花
            continue
        if czgc[l_czgc-i-2]['wz'] != wz:
            return False
        return cz_ in last_cz
    return False


def score_end(pjxx, wz, pmsl, extrascore):
    jfs = extrascore
    jfs += pmsl[pjxx['fp'] - 1] * 10
    pjxx['tjfs'] += ', 番牌:' + str(pmsl[pjxx['fp'] - 1] * 10)
    if pjxx['wjxx'][wz]['mpq_mg'] > 0 or pjxx['wjxx'][wz]['mpq_ag'] > 0:
        # 杠 +50
        jfs += 50
        pjxx['tjfs'] += ', 我方杠:50'
        mpq_pm = pjxx['wjxx'][wz]['mpq_pmsz'][0]
        if mpq_pm == pjxx['fp']:
            jfs += 10
            pjxx['tjfs'] += ', 杠+番:10'
        if mpq_pm >= 10:
            jfs += 10
            pjxx['tjfs'] += ', 杠+字:10'
        else:
            jfs += mpq_pm
            pjxx['tjfs'] += ', 杠+序:10'
        # 杠上开花
        bool_g = czgc_last(pjxx['czgc'], wz, [2, 3, 4], 6)
        if bool_g == 1:
            jfs += 30
            pjxx['tjfs'] += ', 杠上开花:30'
    dfwz = 1 - wz
    if pjxx['wjxx'][dfwz]['mpq_mg'] > 0 or pjxx['wjxx'][dfwz]['mpq_ag'] > 0:
        # 对方杠 -50
        jfs -= 50
        pjxx['tjfs'] += ', 对方杠:-50'

    if pjxx['zj'] != wz and len(pjxx['wjxx'][0]['mpq_pmsz']) == 0 and len(pjxx['wjxx'][1]['mpq_pmsz']) == 0:
        if len(pjxx['wjxx'][wz]['ycp']) == 0:
            # 地胡
            jfs += 40
            pjxx['tjfs'] += ', 地胡:40'
    if pjxx['wjxx'][wz]['btzt'] == 1:
        # 天听
        jfs += 20
        pjxx['tjfs'] += ', 天听:20'
    elif pjxx['wjxx'][wz]['btzt'] == 2:
        # 报听
        jfs += 10
        pjxx['tjfs'] += ', 报听:10'
    # 补花
    jfs += pjxx['wjxx'][wz]['bhs'] * 20
    pjxx['tjfs'] += ', 我方补花:' + str(pjxx['wjxx'][wz]['bhs'] * 20)
    jfs -= pjxx['wjxx'][1 - wz]['bhs'] * 20
    pjxx['tjfs'] += ', 对方补花:' + str(-pjxx['wjxx'][1 - wz]['bhs'] * 20)
    # 五张加番
    n_ = len(pjxx['zpm'])
    if n_ > 5:
        n_ = 5
    elif n_ == 0:
        return jfs
    fs = 0
    for pm in pjxx['zpm'][:n_]:
        if pm == 13:
            fs += 1
        elif pmsl[pm - 1] > 0:
            fs += 1
    if fs == 0:
        return jfs
    choice = [10, 25, 45, 70, 100]
    jfs += choice[fs - 1]
    pjxx['tjfs'] += ', 奖花:' + str(choice[fs - 1])
    return jfs


def pjxx_end(pjxx, yjwz, fs, jslx, jsbz):
    # 流局
    if jslx == 0:
        pjxx['loser_zfz'] = -fs
    else:
        pjxx['loser_zfz'] = pm_zfz(pjxx, 1 - yjwz)
    pjxx['jslx'] = jslx
    pjxx['jsbz'] = jsbz
    pjxx['jsyj'] = yjwz
    pjxx['jsfs'] = [0, 0]
    pjxx['jsfs'][yjwz] = fs
    pjxx['jsfs'][1-yjwz] = -fs


def pjxx_bp(pjxx, wz):
    pm_bp = pjxx['zpm'][0]
    pjxx['wjxx'][wz]['sysl'][pm_bp - 1] -= 1
    while pm_bp == 13:
        pjxx['wjxx'][1-wz]['sysl'][12] -= 1
        add_czgc(pjxx, wz, 7, 13, '补花')
        pjxx['wjxx'][wz]['bhs'] += 1
        pjxx['zpm'].pop(0)
        pjxx['syps'] -= 1
        if pjxx['syps'] <= 0:
            pjxx_end(pjxx, wz, score_buhua_gang(pjxx, wz), 0, '流局')
            return True, pm_bp
        pm_bp = pjxx['zpm'][0]
        pjxx['wjxx'][wz]['sysl'][pm_bp - 1] -= 1
    pjxx['zpm'].pop(0)
    pjxx['syps'] -= 1
    pjxx['fbh_syps'] -= 1
    pjxx['wjxx'][wz]['pm'].append(pm_bp)
    pjxx['wjxx'][wz]['pmsl'][pm_bp - 1] += 1
    add_czgc(pjxx, wz, 6, pm_bp, '补牌')
    return False, pm_bp


def pjxx_tiaoshi(pjxx, wz, pm_bp):
    if wz == 1 and pm_bp == 0 and pjxx['wjxx'][wz]['pm'] == [10, 6, 3, 9, 3] \
            and pjxx['wjxx'][1-wz]['pm'] == [5, 8, 11, 1] and len(pjxx['czgc']) == 0:
        return True
    else:
        return False


def pjxx_match(pjxx, wz, alpha, beta):
    if pjxx is None:
        return 3
    if beta is not None:
        if pjxx['jsfs'][wz] > beta['jsfs'][wz]:
            return 1
    if alpha is None:
        return 2
    if pjxx['jsfs'][wz] > alpha['jsfs'][wz]:
        return 2
    elif pjxx['jsfs'][wz] == alpha['jsfs'][wz]:
        if pjxx['loser_zfz'] > alpha['loser_zfz']:
            return 2
    return 3


def pjxx_match_old(pjxx, wz, best_pjxx, alpha, beta):
    if pjxx is None:
        return best_pjxx, alpha, False
    # best_pjxx
    if best_pjxx is None:
        best_pjxx = copy.deepcopy(pjxx)
    if pjxx['jsfs'][wz] > best_pjxx['jsfs'][wz]:
        best_pjxx = copy.deepcopy(pjxx)
    elif pjxx['jsfs'][wz] == best_pjxx['jsfs'][wz]:
        if pjxx['loser_zfz'] > best_pjxx['loser_zfz']:
            best_pjxx = copy.deepcopy(pjxx)
    # alpha
    if alpha is None:
        alpha = copy.deepcopy(pjxx)
    if pjxx['jsfs'][wz] > alpha['jsfs'][wz]:
        alpha = copy.deepcopy(pjxx)
    elif pjxx['jsfs'][wz] == alpha['jsfs'][wz]:
        if pjxx['loser_zfz'] > alpha['loser_zfz']:
            alpha = copy.deepcopy(pjxx)
    if beta is not None:
        if pjxx['jsfs'][wz] > beta['jsfs'][wz]:
            return best_pjxx, alpha, True
    return best_pjxx, alpha, False


def wjxx_zjpj(pjxx_):
    if pjxx_['jsyj'] != 1:
        return
    # wjxx['qbpj'][1].append(pjxx_)
    if wjxx['fs'][0] == 1 and wjxx['fs'][1] == 0:
        if not wjxx['zjpj'][1]:
            wjxx['zjpj'][1] = [pjxx_]
        elif pjxx_['jsfs'][1] > wjxx['zjpj'][1][0]['jsfs'][1]:
            wjxx['zjpj'][1] = [pjxx_]
        elif pjxx_['jsfs'][1] == wjxx['zjpj'][1][0]['jsfs'][1] and\
                pjxx_['fbh_syps'] == wjxx['zjpj'][1][0]['fbh_syps']:
            wjxx['zjpj'][1].append(pjxx_)
        if wjxx['zkpj'][1] is None:
            wjxx['zkpj'][1] = pjxx_
        elif pjxx_['fbh_syps'] > wjxx['zkpj'][1]['fbh_syps']:
            wjxx['zkpj'][1] = pjxx_
        elif pjxx_['fbh_syps'] == wjxx['zkpj'][1]['fbh_syps'] and\
                pjxx_['jsfs'][1] > wjxx['zkpj'][1]['jsfs'][1]:
            wjxx['zkpj'][1] = pjxx_


def xunhuancaozuo(pjxx, wz, dfcp, alpha, beta):
    # if pjxx['czgc'] == [{'wz': 1, 'cz': 1, 'pm': 11, 'bz': '出牌', 'zjpm': [2, 3, 12, 10]}, {'wz': 0, 'cz': 7, 'pm': 13, 'bz': '补花', 'zjpm': [1, 11, 8, 11]}, {'wz': 0, 'cz': 6, 'pm': 3, 'bz': '补牌', 'zjpm': [1, 11, 8, 11, 3]}, {'wz': 0, 'cz': 1, 'pm': 8, 'bz': '出牌', 'zjpm': [1, 11, 11, 3]}, {'wz': 1, 'cz': 6, 'pm': 2, 'bz': '补牌', 'zjpm': [2, 3, 12, 10, 2]}, {'wz': 1, 'cz': 1, 'pm': 10, 'bz': '出牌', 'zjpm': [2, 3, 12, 2]}, {'wz': 0, 'cz': 7, 'pm': 13, 'bz': '补花', 'zjpm': [1, 11, 11, 3]}, {'wz': 0, 'cz': 6, 'pm': 11, 'bz': '补牌', 'zjpm': [1, 11, 11, 3, 11]}, {'wz': 0, 'cz': 1, 'pm': 1, 'bz': '出牌', 'zjpm': [11, 11, 3, 11]}, {'wz': 1, 'cz': 6, 'pm': 5, 'bz': '补牌', 'zjpm': [2, 3, 12, 2, 5]}, {'wz': 1, 'cz': 1, 'pm': 12, 'bz': '出牌', 'zjpm': [2, 3, 2, 5]}, {'wz': 0, 'cz': 6, 'pm': 2, 'bz': '补牌', 'zjpm': [11, 11, 3, 11, 2]}, {'wz': 0, 'cz': 1, 'pm': 2, 'bz': '出牌', 'zjpm': [11, 11, 3, 11]}]:
    #     print(dfcp, pjxx['wjxx'][wz]['pm'])
    #     print(alpha)
    #     print(beta)
    #     show_detail = True
    # else:
    #     show_detail = False
    best_pjxx = None
    dfwz = 1 - wz
    if dfcp != 0:
        # 对方出牌:明杠
        if pjxx['wjxx'][wz]['pmsl'][dfcp - 1] == 3:
            pjxx_2 = copy.deepcopy(pjxx)
            pjxx_2['wjxx'][wz]['pmsl'][dfcp - 1] -= 3
            pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
            for j in range(3):
                pjxx_2['wjxx'][wz]['pm'].remove(dfcp)
                pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
            pjxx_2['wjxx'][wz]['mpq_mg'] = dfcp
            pjxx_2['wjxx'][dfwz]['sysl'][dfcp - 1] -= 3
            add_czgc(pjxx_2, wz, 4, dfcp, '明杠')
            pjxx_r = xunhuancaozuo(pjxx_2, wz, 0, alpha, beta)
            best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
            if bool_rt:
                return best_pjxx
            if wjxx['fs'][wz] == 1:
                return best_pjxx
        # 对方出牌:碰
        if pjxx['wjxx'][wz]['pmsl'][dfcp - 1] >= 2 and pjxx['wjxx'][wz]['btzt'] == 0:
            bool_p_ = True
            if wjxx['fs'][wz] == 1:
                bool_p_ = getnp_p(pjxx, wz, dfcp)
            if bool_p_:
                pjxx_2 = copy.deepcopy(pjxx)
                pjxx_2['wjxx'][wz]['pmsl'][dfcp - 1] -= 2
                pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
                for j in range(2):
                    pjxx_2['wjxx'][wz]['pm'].remove(dfcp)
                    pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
                pjxx_2['wjxx'][wz]['mpq_p'] = dfcp
                pjxx_2['wjxx'][dfwz]['sysl'][dfcp - 1] -= 2
                add_czgc(pjxx_2, wz, 5, dfcp, '碰')
                # print("碰", pjxx_2)
                pjxx_r = xunhuancaozuo(pjxx_2, wz, 0, alpha, beta)
                best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
                if bool_rt:
                    return best_pjxx
                if wjxx['fs'][wz] == 1:
                    return best_pjxx
    pm_bp = 0
    if len(pjxx['wjxx'][wz]['pm']) % 3 != 2:
        # 需要补牌
        if pjxx['syps'] <= 0:
            # print('流局')
            pjxx_end(pjxx, wz, score_buhua_gang(pjxx, wz), 0, '流局')
            # wjxx_zjpj(pjxx)
            best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx, wz, best_pjxx, alpha, beta)
            return best_pjxx
        bool_rt, pm_bp = pjxx_bp(pjxx, wz)
        if bool_rt:
            # 流局
            # wjxx_zjpj(pjxx)
            best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx, wz, best_pjxx, alpha, beta)
            return best_pjxx
    fs = pmsl_is_hu(pjxx['wjxx'][wz]['pmsl'], pjxx['wjxx'][wz]['mpq_pmsz'])
    if fs > 0:
        pjxx['tjfs'] = '基础分数:' + str(fs)
        if pjxx['zj'] == wz and len(pjxx['wjxx'][wz]['ycp']) == 0 and len(pjxx['wjxx'][wz]['mpq_pmsz']) == 0:
            # 天胡
            pjxx['tjfs'] += ', 天胡:80'
            fs += score_end(pjxx, wz, pjxx['wjxx'][wz]['pmsl'], 80)
        else:
            # 自摸
            pjxx['tjfs'] += ', 自摸:10'
            fs += score_end(pjxx, wz, pjxx['wjxx'][wz]['pmsl'], 10)
        pjxx_end(pjxx, wz, fs, 1, '自摸')
        # print("自摸", pjxx['czgc'])
        # wjxx_zjpj(pjxx)
        best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx, wz, best_pjxx, alpha, beta)
        return best_pjxx
    if len(pjxx['wjxx'][wz]['mpq_pmsz']) == 0:
        for i in range(12):
            if pjxx['wjxx'][wz]['pmsl'][i] == 4:
                # 暗杠
                pjxx_ = copy.deepcopy(pjxx)
                pjxx_['wjxx'][wz]['pmsl'][i] = 0
                for j in range(4):
                    pjxx_['wjxx'][wz]['pm'].remove(i + 1)
                    pjxx_['wjxx'][wz]['mpq_pmsz'].append(i + 1)
                pjxx_['wjxx'][wz]['mpq_ag'] = i + 1
                add_czgc(pjxx_, wz, 2, i + 1, '暗杠')
                pjxx_r = xunhuancaozuo(pjxx_, wz, 0, alpha, beta)
                best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
                if bool_rt:
                    return best_pjxx
                if wjxx['fs'][wz] == 1:
                    return best_pjxx
                break
    elif pjxx['wjxx'][wz]['mpq_p'] > 0:
        pm_cz = pjxx['wjxx'][wz]['mpq_p']
        if pjxx['wjxx'][wz]['pmsl'][pm_cz - 1] == 1:
            # 明杠 碰后杠
            pjxx_ = copy.deepcopy(pjxx)
            pjxx_['wjxx'][wz]['pmsl'][pm_cz - 1] = 0
            pjxx_['wjxx'][wz]['pm'].remove(pm_cz)
            pjxx_['wjxx'][wz]['mpq_pmsz'].append(pm_cz)
            pjxx_['wjxx'][wz]['mpq_mg'] = pm_cz
            pjxx_['wjxx'][wz]['mpq_p'] = 0
            pjxx_['wjxx'][1 - wz]['sysl'][pm_cz - 1] -= 1
            add_czgc(pjxx_, wz, 3, pm_cz, '碰后杠')
            pjxx_r = xunhuancaozuo(pjxx_, wz, 0, alpha, beta)
            best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
            if bool_rt:
                return best_pjxx
            if wjxx['fs'][wz] == 1:
                return best_pjxx
    if pjxx['wjxx'][wz]['btzt'] >= 1 and pm_bp > 0:
        # 报听
        cplb, bss = [pm_bp], [1]
        n_ = 1
    else:
        if wjxx['fs'][wz] == 1:
            np_cp = getnp_cp(pjxx, wz)
            np_bss = []
            for np_cp_pm in np_cp:
                np_pmsl = pjxx['wjxx'][wz]['pmsl'][:]
                np_pmsl[np_cp_pm-1] -= 1
                np_bss_ = pmsl_get_bss(
                    np_pmsl,
                    pjxx['wjxx'][wz]['mpq_pmsz'])
                np_bss.append(np_bss_)
            cplb, bss = np_cp[:], np_bss[:]
            n_ = len(cplb)
            # print(wz, cplb, bss, pjxx['wjxx'][wz], pjxx['czgc'])
        else:
            cplb, bss = best_cp(pjxx, wz)
            n_ = len(cplb)
            if n_ > 2:
                n_ = 2
    for xh_ in range(n_):
        pm_cp = cplb[xh_]
        if pjxx['wjxx'][wz]['pmsl'][pm_cp - 1] == 0:
            print("error cp", pjxx, wz, cplb, n_)
            continue
        # if xh_ == 1:
        #     if bss[xh_] > bss[0]:
        #         continue
        # 出牌
        pjxx_ = copy.deepcopy(pjxx)
        pjxx_['wjxx'][wz]['pmsl'][pm_cp - 1] -= 1
        pjxx_['wjxx'][wz]['pm'].remove(pm_cp)
        pjxx_['wjxx'][wz]['ycp'].append(pm_cp)
        pjxx_['wjxx'][1 - wz]['sysl'][pm_cp - 1] -= 1
        if pjxx_['wjxx'][wz]['btzt'] == 0:
            pjxx_['wjxx'][wz]['ycp_tq'].append(pm_cp)
        else:
            pjxx_['wjxx'][wz]['ycp_th'].append(pm_cp)
        if pjxx_['wjxx'][1 - wz]['btzt'] == 0:
            pjxx_['wjxx'][wz]['ycp_tq_df'].append(pm_cp)
        else:
            pjxx_['wjxx'][wz]['ycp_th_df'].append(pm_cp)
        add_czgc(pjxx_, wz, 1, pm_cp, '出牌')
        pjxx_['wjxx'][wz]['dcpm'], pjxx_['wjxx'][wz]['dcfs'], bool_rt = pmsl_get_dcp(
            pjxx_['wjxx'][wz]['pmsl'],
            pjxx_['wjxx'][wz]['mpq_pmsz'])
        # print('位置', wz, '出牌', pm_cp, pjxx_['wjxx'][wz]['pm'], pjxx_['wjxx'][wz]['pmsl'])
        # 判断是否点炮
        if pm_cp in pjxx_['wjxx'][dfwz]['dcpm']:
            sy_ = pjxx_['wjxx'][dfwz]['dcpm'].index(pm_cp)
            cpfs = pjxx_['wjxx'][dfwz]['dcfs'][sy_]
            # 判断其他番数
            pmsl_ = pjxx_['wjxx'][dfwz]['pmsl'][:]
            pmsl_[pm_cp - 1] += 1
            pjxx_['tjfs'] = '基础分数:' + str(cpfs)
            cpfs += score_end(pjxx_, dfwz, pmsl_, 0)
            pjxx_r = copy.deepcopy(pjxx_)
            pjxx_end(pjxx_r, dfwz, cpfs, 2, '点炮')
            best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
            # if pjxx_tiaoshi(pjxx, wz, pm_bp):
            #     print(int_rt, pjxx_r['jsfs'][wz], pjxx_r)
            # wjxx_zjpj(pjxx_r)
            if bool_rt:
                return best_pjxx
            # print('结束', dfwz, '点炮', pm_cp, cpfs, pjxx_['wjxx'][dfwz]['pm'])
            continue
        # 报听
        if bss[xh_] == 1 and pjxx_['wjxx'][wz]['btzt'] == 0:
            pjxx_2 = copy.deepcopy(pjxx_)
            if not bool_rt:
                print(pjxx_2['wjxx'][wz]['pmsl'], pjxx_2['wjxx'][wz]['dcpm'], pjxx_2['wjxx'][wz]['dcfs'], bool_rt,
                      pjxx_2['wjxx'][wz]['mpq_pmsz'])
            else:
                if wz == pjxx_['zj'] and len(pjxx_['wjxx'][wz]['ycp']) == 1 and \
                        len(pjxx_['wjxx'][dfwz]['ycp']) == 0 and \
                        len(pjxx_['wjxx'][wz]['mpq_pmsz']) == 0:
                    btlx = 2
                else:
                    btlx = 3
                if wjxx['fs'][wz] == 1:
                    bool_bt_ = getnp_bt(pjxx_, wz, pm_cp, btlx)
                    if bool_bt_:
                        if btlx == 2:
                            pjxx_['wjxx'][wz]['btzt'] = 1
                            add_czgc(pjxx_, wz, 9, 0, '庄家天听')
                        else:
                            pjxx_['wjxx'][wz]['btzt'] = 2
                            add_czgc(pjxx_, wz, 10, 0, '报听')
                        n = len(pjxx_['wjxx'][wz]['ycp_tq'])
                        if n >= 1:
                            pjxx_['wjxx'][wz]['ycp_th'].append(pjxx_['wjxx'][wz]['ycp_tq'][-1])
                            pjxx_['wjxx'][wz]['ycp_tq'].pop()
                else:
                    if btlx == 2:
                        pjxx_2['wjxx'][wz]['btzt'] = 1
                        add_czgc(pjxx_2, wz, 9, 0, '庄家天听')
                    else:
                        pjxx_2['wjxx'][wz]['btzt'] = 2
                        add_czgc(pjxx_2, wz, 10, 0, '报听')
                    n = len(pjxx_2['wjxx'][wz]['ycp_tq'])
                    if n >= 1:
                        pjxx_2['wjxx'][wz]['ycp_th'].append(pjxx_2['wjxx'][wz]['ycp_tq'][-1])
                        pjxx_2['wjxx'][wz]['ycp_tq'].pop()
                    pjxx_r = xunhuancaozuo(pjxx_2, dfwz, pm_cp, beta, alpha)
                    best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
                    if bool_rt:
                        return best_pjxx
        pjxx_r = xunhuancaozuo(pjxx_, dfwz, pm_cp, beta, alpha)
        # print('位置', dfwz, '补牌', pm_bp, pjxx_['wjxx'][dfwz]['pm'], pjxx_['wjxx'][dfwz]['pmsl'], fs)
        best_pjxx, alpha, bool_rt = pjxx_match_old(pjxx_r, wz, best_pjxx, alpha, beta)
        if bool_rt:
            return best_pjxx
    return best_pjxx


def wjxx_zjpj_1_0(pjxx_r):
    if wjxx['zjpj'] is None:
        wjxx['zjpj'] = pjxx_r
    elif pjxx_r['jsfs'][1] > wjxx['zjpj']['jsfs'][1]:
        wjxx['zjpj'] = pjxx_r
    elif pjxx_r['jsfs'][1] == wjxx['zjpj']['jsfs'][1]:
        if pjxx_r['loser_zfz'] > wjxx['zjpj']['loser_zfz']:
            wjxx['zjpj'] = pjxx_r


def xunhuancaozuo_1_0(pjxx, wz, dfcp):
    # 位置0 tf 位置1 switch
    dfwz = 1 - wz
    if dfcp != 0:
        # 对方出牌:明杠
        if pjxx['wjxx'][wz]['pmsl'][dfcp - 1] == 3:
            pjxx_2 = copy.deepcopy(pjxx)
            pjxx_2['wjxx'][wz]['pmsl'][dfcp - 1] -= 3
            pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
            for j in range(3):
                pjxx_2['wjxx'][wz]['pm'].remove(dfcp)
                pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
            pjxx_2['wjxx'][wz]['mpq_mg'] = dfcp
            pjxx_2['wjxx'][dfwz]['sysl'][dfcp - 1] -= 3
            add_czgc(pjxx_2, wz, 4, dfcp, '明杠')
            pjxx_r = xunhuancaozuo_1_0(pjxx_2, wz, 0)
            wjxx_zjpj_1_0(pjxx_r)
            if wjxx['fs'][wz] == 1:
                return wjxx['zjpj']
        # 对方出牌:碰
        if pjxx['wjxx'][wz]['pmsl'][dfcp - 1] >= 2 and pjxx['wjxx'][wz]['btzt'] == 0:
            bool_p_ = True
            if wjxx['fs'][wz] == 1:
                bool_p_ = getnp_p(pjxx, wz, dfcp)
            if bool_p_:
                pjxx_2 = copy.deepcopy(pjxx)
                pjxx_2['wjxx'][wz]['pmsl'][dfcp - 1] -= 2
                pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
                for j in range(2):
                    pjxx_2['wjxx'][wz]['pm'].remove(dfcp)
                    pjxx_2['wjxx'][wz]['mpq_pmsz'].append(dfcp)
                pjxx_2['wjxx'][wz]['mpq_p'] = dfcp
                pjxx_2['wjxx'][dfwz]['sysl'][dfcp - 1] -= 2
                add_czgc(pjxx_2, wz, 5, dfcp, '碰')
                # print("碰", pjxx_2)
                pjxx_r = xunhuancaozuo_1_0(pjxx_2, wz, 0)
                wjxx_zjpj_1_0(pjxx_r)
                if wjxx['fs'][wz] == 1:
                    return wjxx['zjpj']
    pm_bp = 0
    if len(pjxx['wjxx'][wz]['pm']) % 3 != 2:
        # 需要补牌
        if pjxx['syps'] <= 0:
            # print('流局')
            pjxx_end(pjxx, wz, score_buhua_gang(pjxx, wz), 0, '流局')
            wjxx_zjpj_1_0(pjxx)
            return wjxx['zjpj']
        bool_rt, pm_bp = pjxx_bp(pjxx, wz)
        if bool_rt:
            # 流局
            wjxx_zjpj_1_0(pjxx)
            return wjxx['zjpj']
    fs = pmsl_is_hu(pjxx['wjxx'][wz]['pmsl'], pjxx['wjxx'][wz]['mpq_pmsz'])
    # print(fs)
    if fs > 0:
        pjxx['tjfs'] = '基础分数:' + str(fs)
        if pjxx['zj'] == wz and len(pjxx['wjxx'][wz]['ycp']) == 0 and \
                len(pjxx['wjxx'][wz]['mpq_pmsz']) == 0:
            # 天胡
            pjxx['tjfs'] += ', 天胡:80'
            fs += score_end(pjxx, wz, pjxx['wjxx'][wz]['pmsl'], 80)
        else:
            # 自摸
            pjxx['tjfs'] += ', 自摸:10'
            fs += score_end(pjxx, wz, pjxx['wjxx'][wz]['pmsl'], 10)
        pjxx_end(pjxx, wz, fs, 1, '自摸')
        # print("自摸", pjxx['czgc'])
        wjxx_zjpj_1_0(pjxx)
        return wjxx['zjpj']
    if len(pjxx['wjxx'][wz]['mpq_pmsz']) == 0:
        for i in range(12):
            if pjxx['wjxx'][wz]['pmsl'][i] == 4:
                # 暗杠
                pjxx_ = copy.deepcopy(pjxx)
                pjxx_['wjxx'][wz]['pmsl'][i] = 0
                for j in range(4):
                    pjxx_['wjxx'][wz]['pm'].remove(i + 1)
                    pjxx_['wjxx'][wz]['mpq_pmsz'].append(i + 1)
                pjxx_['wjxx'][wz]['mpq_ag'] = i + 1
                add_czgc(pjxx_, wz, 2, i + 1, '暗杠')
                pjxx_r = xunhuancaozuo_1_0(pjxx_, wz, 0)
                wjxx_zjpj_1_0(pjxx_r)
                if wjxx['fs'][wz] == 1:
                    return wjxx['zjpj']
                break
    elif pjxx['wjxx'][wz]['mpq_p'] > 0:
        pm_cz = pjxx['wjxx'][wz]['mpq_p']
        if pjxx['wjxx'][wz]['pmsl'][pm_cz - 1] == 1:
            # 明杠 碰后杠
            pjxx_ = copy.deepcopy(pjxx)
            pjxx_['wjxx'][wz]['pmsl'][pm_cz - 1] = 0
            pjxx_['wjxx'][wz]['pm'].remove(pm_cz)
            pjxx_['wjxx'][wz]['mpq_pmsz'].append(pm_cz)
            pjxx_['wjxx'][wz]['mpq_mg'] = pm_cz
            pjxx_['wjxx'][wz]['mpq_p'] = 0
            pjxx_['wjxx'][1 - wz]['sysl'][pm_cz - 1] -= 1
            add_czgc(pjxx_, wz, 3, pm_cz, '碰后杠')
            pjxx_r = xunhuancaozuo_1_0(pjxx_, wz, 0)
            wjxx_zjpj_1_0(pjxx_r)
            if wjxx['fs'][wz] == 1:
                return wjxx['zjpj']
    if pjxx['wjxx'][wz]['btzt'] >= 1 and pm_bp > 0:
        # 报听
        cplb, bss = [pm_bp], [1]
        n_ = 1
    else:
        if wjxx['fs'][wz] == 1:
            np_cp = getnp_cp(pjxx, wz)
            np_bss = []
            for np_cp_pm in np_cp:
                np_pmsl = pjxx['wjxx'][wz]['pmsl'][:]
                np_pmsl[np_cp_pm-1] -= 1
                np_bss_ = pmsl_get_bss(
                    np_pmsl,
                    pjxx['wjxx'][wz]['mpq_pmsz'])
                np_bss.append(np_bss_)
            cplb, bss = np_cp[:], np_bss[:]
            n_ = len(cplb)
            # print(wz, cplb, bss, pjxx['wjxx'][wz], pjxx['czgc'])
        else:
            cplb, bss = best_cp(pjxx, wz)
            n_ = len(cplb)
            if n_ > 2:
                n_ = 2
    for xh_ in range(n_):
        pm_cp = cplb[xh_]
        if pjxx['wjxx'][wz]['pmsl'][pm_cp - 1] == 0:
            print("error cp", pjxx, wz, cplb, n_)
            continue
        # if xh_ == 1:
        #     if bss[xh_] > bss[0]:
        #         continue
        # 出牌
        pjxx_ = copy.deepcopy(pjxx)
        pjxx_['wjxx'][wz]['pmsl'][pm_cp - 1] -= 1
        pjxx_['wjxx'][wz]['pm'].remove(pm_cp)
        pjxx_['wjxx'][wz]['ycp'].append(pm_cp)
        pjxx_['wjxx'][1 - wz]['sysl'][pm_cp - 1] -= 1
        if pjxx_['wjxx'][wz]['btzt'] == 0:
            pjxx_['wjxx'][wz]['ycp_tq'].append(pm_cp)
        else:
            pjxx_['wjxx'][wz]['ycp_th'].append(pm_cp)
        if pjxx_['wjxx'][1 - wz]['btzt'] == 0:
            pjxx_['wjxx'][wz]['ycp_tq_df'].append(pm_cp)
        else:
            pjxx_['wjxx'][wz]['ycp_th_df'].append(pm_cp)
        add_czgc(pjxx_, wz, 1, pm_cp, '出牌')
        pjxx_['wjxx'][wz]['dcpm'], pjxx_['wjxx'][wz]['dcfs'], bool_rt = pmsl_get_dcp(
            pjxx_['wjxx'][wz]['pmsl'],
            pjxx_['wjxx'][wz]['mpq_pmsz'])
        # print('位置', wz, '出牌', pm_cp, pjxx_['wjxx'][wz]['pm'], pjxx_['wjxx'][wz]['pmsl'])
        # 判断是否点炮
        if pm_cp in pjxx_['wjxx'][dfwz]['dcpm']:
            sy_ = pjxx_['wjxx'][dfwz]['dcpm'].index(pm_cp)
            cpfs = pjxx_['wjxx'][dfwz]['dcfs'][sy_]
            # 判断其他番数
            pmsl_ = pjxx_['wjxx'][dfwz]['pmsl'][:]
            pmsl_[pm_cp - 1] += 1
            pjxx_['tjfs'] = '基础分数:' + str(cpfs)
            cpfs += score_end(pjxx_, dfwz, pmsl_, 0)
            pjxx_r = copy.deepcopy(pjxx_)
            pjxx_end(pjxx_r, dfwz, cpfs, 2, '点炮')
            wjxx_zjpj_1_0(pjxx_r)
            # print('结束', dfwz, '点炮', pm_cp, cpfs, pjxx_['wjxx'][dfwz]['pm'])
            continue
        # 报听
        if bss[xh_] == 1 and pjxx_['wjxx'][wz]['btzt'] == 0:
            pjxx_2 = copy.deepcopy(pjxx_)
            if not bool_rt:
                print(pjxx_2['wjxx'][wz]['pmsl'], pjxx_2['wjxx'][wz]['dcpm'], pjxx_2['wjxx'][wz]['dcfs'], bool_rt,
                      pjxx_2['wjxx'][wz]['mpq_pmsz'])
            else:
                if wz == pjxx_['zj'] and len(pjxx_['wjxx'][wz]['ycp']) == 1 and \
                        len(pjxx_['wjxx'][dfwz]['ycp']) == 0 and \
                        len(pjxx_['wjxx'][wz]['mpq_pmsz']) == 0:
                    btlx = 2
                else:
                    btlx = 3
                if wjxx['fs'][wz] == 1:
                    bool_bt_ = getnp_bt(pjxx_, wz, pm_cp, btlx)
                    if bool_bt_:
                        if btlx == 2:
                            pjxx_['wjxx'][wz]['btzt'] = 1
                            add_czgc(pjxx_, wz, 9, 0, '庄家天听')
                        else:
                            pjxx_['wjxx'][wz]['btzt'] = 2
                            add_czgc(pjxx_, wz, 10, 0, '报听')
                        n = len(pjxx_['wjxx'][wz]['ycp_tq'])
                        if n >= 1:
                            pjxx_['wjxx'][wz]['ycp_th'].append(pjxx_['wjxx'][wz]['ycp_tq'][-1])
                            pjxx_['wjxx'][wz]['ycp_tq'].pop()
                else:
                    if btlx == 2:
                        pjxx_2['wjxx'][wz]['btzt'] = 1
                        add_czgc(pjxx_2, wz, 9, 0, '庄家天听')
                    else:
                        pjxx_2['wjxx'][wz]['btzt'] = 2
                        add_czgc(pjxx_2, wz, 10, 0, '报听')
                    n = len(pjxx_2['wjxx'][wz]['ycp_tq'])
                    if n >= 1:
                        pjxx_2['wjxx'][wz]['ycp_th'].append(pjxx_2['wjxx'][wz]['ycp_tq'][-1])
                        pjxx_2['wjxx'][wz]['ycp_tq'].pop()
                    pjxx_r = xunhuancaozuo_1_0(pjxx_2, dfwz, pm_cp)
                    wjxx_zjpj_1_0(pjxx_r)
        pjxx_r = xunhuancaozuo_1_0(pjxx_, dfwz, pm_cp)
        # print('位置', dfwz, '补牌', pm_bp, pjxx_['wjxx'][dfwz]['pm'], pjxx_['wjxx'][dfwz]['pmsl'], fs)
        wjxx_zjpj_1_0(pjxx_r)
    return wjxx['zjpj']


def pjxx_xh(no, tjxx_, pjxx_crcs, wjxx_0_, wjxx_1_):
    # while not stop_thread:
    wjxx.update({'zjpj': [[], []], 'zkpj': [None, None], 'qbpj': [[], []]})
    strStartDateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    startSeconds = time.mktime(time.strptime(strStartDateTime, "%Y-%m-%d %X"))
    if pjxx_crcs is not None:
        pjxx = copy.deepcopy(pjxx_crcs)
    else:
        pjxx = pjxx_csh()
    '''
    pjxx = {'zpm': [9, 10, 10, 1, 13, 7, 12, 13, 8, 6, 12, 3, 4, 1, 5, 2, 2, 2, 7, 9, 4, 5, 11, 10, 9, 2,
                    1, 4, 3, 7, 12, 1, 4, 9, 12, 3, 5, 6, 7, 13, 6, 8, 13],
            'wjxx': [
                {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0,
                 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [],
                 'ycp_th_df': [], 'bhs': 0, 'pm': [3, 11, 6, 8, 5],
                 'pmsl': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
                 'sysl': [4, 4, 3, 4, 3, 3, 4, 3, 4, 4, 3, 4]},
                {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0,
                 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [],
                 'ycp_th_df': [], 'bhs': 0, 'pm': [10, 8, 11, 11],
                 'pmsl': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0],
                 'sysl': [4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 2, 4]}],
            'czgc': [], 'loser_zfz': 0, 'fp': 4, 'zj': 0, 'syps': 43, 'fbh_syps': 39}
    '''
    # pjxx = {'zpm': [4, 10, 5, 8, 7, 11, 13, 4, 2, 8, 13, 11, 2, 4, 3, 13, 7, 9, 7, 1, 11, 1, 8, 11, 10, 6, 12, 12, 12, 9, 9, 4, 6, 6, 10, 2, 10, 8, 3, 7, 9, 2], 'wjxx': [{'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0, 'pm': [1, 12, 6, 3, 3], 'pmsl': [1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], 'sysl': [3, 4, 2, 4, 4, 3, 4, 4, 4, 4, 4, 3]}, {'dcpm': [1], 'dcfs': [47], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 1, 'pm': [5, 5, 1, 5], 'pmsl': [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], 'sysl': [3, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4]}], 'czgc': [{'wz': 1, 'fs': 3, 'lx': 2, 'pm': 13, 'bz': '补花', 'zjpm': [5, 5, 1]}], 'loser_zfz': 0, 'fp': 5, 'zj': 0, 'syps': 42, 'fbh_syps': 39}
    # pjxx = {'zpm': [1, 13, 13, 5, 1, 12, 8, 4, 2, 5, 10, 3, 6, 12, 2, 8, 11, 7, 2, 1, 10, 4, 13, 9, 8, 5, 6, 11, 5, 11, 10, 3, 7, 9, 8, 1, 13, 3, 11, 3, 6, 10, 12], 'wjxx': [{'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0, 'pm': [4, 9, 2, 12], 'pmsl': [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0], 'sysl': [4, 3, 4, 3, 4, 4, 4, 4, 3, 4, 4, 3, 4]}, {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0, 'pm': [4, 9, 7, 6, 7], 'pmsl': [0, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0], 'sysl': [4, 4, 4, 3, 4, 3, 2, 4, 3, 4, 4, 4, 4]}], 'czgc': [], 'loser_zfz': 0, 'fp': 8, 'zj': 1, 'syps': 43, 'fbh_syps': 39}
    fs_tt, pjxx_2 = 0, None
    dfwz = 1 - pjxx['zj']
    pjxx['wjxx'][dfwz]['dcpm'], pjxx['wjxx'][dfwz]['dcfs'], bool_rt = pmsl_get_dcp(pjxx['wjxx'][dfwz]['pmsl'], [])
    # print("牌局初始化完毕", no, pjxx['wjxx'][dfwz]['dcpm'], pjxx['wjxx'][dfwz]['dcfs'], bool_rt, pjxx)
    # print(dfwz, wjxx['fs'][dfwz], pjxx['wjxx'][dfwz]['dcpm'], pjxx['wjxx'][dfwz]['dcfs'], bool_rt)
    if bool_rt:
        if wjxx['fs'][dfwz] == 1:
            if getnp_bt(pjxx, dfwz, 0, 1):
                pjxx['wjxx'][dfwz]['btzt'] = 1
                add_czgc(pjxx, dfwz, 8, 0, '闲家天听')
            bool_rt = False
        else:
            pjxx_2 = copy.deepcopy(pjxx)
            pjxx_2['wjxx'][dfwz]['btzt'] = 1
            add_czgc(pjxx_2, dfwz, 8, 0, '闲家天听')
            pjxx_2 = xunhuancaozuo(pjxx_2, pjxx_2['zj'], 0, None, None)
            fs_tt = pjxx_2['jsfs'][pjxx_2['zj']]
            # print("天听枚举完毕", fs_tt, pjxx_2)
    pjxx_3 = xunhuancaozuo(pjxx, pjxx['zj'], 0, None, pjxx_2)
    fs = pjxx_3['jsfs'][pjxx['zj']]
    # print("枚举完毕", fs, pjxx_3)
    if bool_rt:
        if fs_tt < fs:
            fs_best = fs_tt
            pjxx_best = pjxx_2
        elif fs_tt == fs:
            if pjxx_2['loser_zfz'] > pjxx_3['loser_zfz']:
                fs_best = fs_tt
                pjxx_best = pjxx_2
            else:
                fs_best = fs
                pjxx_best = pjxx_3
        else:
            fs_best = fs
            pjxx_best = pjxx_3
    else:
        fs_best = fs
        pjxx_best = pjxx_3
    strEndDateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    endSeconds = time.mktime(time.strptime(strEndDateTime, "%Y-%m-%d %X"))
    # threadLock.acquire()
    # 统计信息
    # tjxx_['zjs'] += 1
    # if fs_best > 0:
    #     if pjxx['zj'] == 0:
    #         tjxx_['win'] += 1
    # else:
    #     if 1 - pjxx['zj'] == 0:
    #         tjxx_['win'] += 1
    # if pjxx['zj'] == 0:
    #     tjxx_['score'] += fs_best
    # else:
    #     tjxx_['score'] -= fs_best
    # print(pjxx_best)
    # mysql_insert(pjxx, pjxx_best)
    usetime = endSeconds - startSeconds
    tjxx_add(pjxx_best, usetime, tjxx_, wjxx_0_, wjxx_1_)
    # threadLock.release()
    # print(no, "最佳牌局", f'耗费时间：{usetime :.0f}秒', pjxx['zj'], fs_best, pjxx_best, pjxx)
    print(no, f'耗费时间：{usetime :.0f}秒')
    '''
    print("全部牌局数组", len(wjxx['qbpj'][1]), wjxx['qbpj'][1])
    print("最佳牌局数组", len(wjxx['zjpj'][1]), wjxx['zjpj'][1])
    print("最快牌局", wjxx['zkpj'][1])
    print("最佳牌局", pjxx_best)
    
    if wjxx['fs'][0] == 1 and wjxx['fs'][1] == 0:
        mysql_insert_pjxx(pjxx, pjxx_best, 'moni_zjpj')
        mysql_insert_pjxx(pjxx, wjxx['zkpj'][1], 'moni_zkpj')
        # mysql_insert_pjxxsz(pjxx, wjxx['zjpj'][1], 'moni_zjpjsz')
        # mysql_insert_pjxxsz(pjxx, wjxx['qbpj'][1], 'moni_qbpj')
    else:
        mysql_insert_pjxx(pjxx, pjxx_best, 'moni_221202')
    '''
    # mysql_insert_pjxx(pjxx, pjxx_best, 'moni_221202')
    return pjxx_best


def tjxx_add(pjxx_end_, usetime, tjxx_, wjxx_0_, wjxx_1_):
    tjxx_['time'] += usetime
    tjxx_['zjs'] += 1
    # 'th':0, 'dh':0, 'xj_tt':0, 'zj_tt':0, 'bt':0, 'p':0, 'mg':0, 'ag':0, 'gskh':0, 'bhs':0,
    #       'dp':0, 'zm':0, 'yj':0, 'zj':0, 'fs':0
    wjxx_update(pjxx_end_['zj'], 'zj', wjxx_0_, wjxx_1_, 1)
    if pjxx_end_['jsbz'] == '点炮':
        wjxx_update(1-pjxx_end_['jsyj'], 'dp', wjxx_0_, wjxx_1_, 1)
    elif pjxx_end_['jsbz'] == '自摸':
        wjxx_update(pjxx_end_['jsyj'], 'zm', wjxx_0_, wjxx_1_, 1)
    elif pjxx_end_['jsbz'] == '流局':
        tjxx_['lj'] += 1
        return
    else:
        tjxx_['error'] += 1
        return
    wjxx_update(pjxx_end_['jsyj'], 'yj', wjxx_0_, wjxx_1_, 1)
    for i in range(2):
        wjxx_update(i, 'bhs', wjxx_0_, wjxx_1_, pjxx_end_['wjxx'][i]['bhs'])
        wjxx_update(i, 'fs', wjxx_0_, wjxx_1_, pjxx_end_['jsfs'][i])
        if pjxx_end_['wjxx'][i]['mpq_p'] > 0:
            wjxx_update(i, 'p', wjxx_0_, wjxx_1_, 1)
        if pjxx_end_['wjxx'][i]['mpq_mg'] > 0:
            wjxx_update(i, 'mg', wjxx_0_, wjxx_1_, 1)
        if pjxx_end_['wjxx'][i]['mpq_ag'] > 0:
            wjxx_update(i, 'ag', wjxx_0_, wjxx_1_, 1)
        if pjxx_end_['wjxx'][i]['btzt'] == 1:
            if pjxx_end_['zj'] == i:
                wjxx_update(i, 'zj_tt', wjxx_0_, wjxx_1_, 1)
            else:
                wjxx_update(i, 'xj_tt', wjxx_0_, wjxx_1_, 1)
        elif pjxx_end_['wjxx'][i]['btzt'] == 2:
            wjxx_update(i, 'bt', wjxx_0_, wjxx_1_, 1)
    if pjxx_end_['tjfs'].find('天胡:80') != -1:
        wjxx_update(pjxx_end_['jsyj'], 'th', wjxx_0_, wjxx_1_, 1)
    if pjxx_end_['tjfs'].find('地胡:40') != -1:
        wjxx_update(pjxx_end_['jsyj'], 'dh', wjxx_0_, wjxx_1_, 1)
    if pjxx_end_['tjfs'].find('杠上开花:30') != -1:
        wjxx_update(pjxx_end_['jsyj'], 'gskh', wjxx_0_, wjxx_1_, 1)


def wjxx_update(wz, key_, wjxx_0_, wjxx_1_, plus_n):
    if wz == 0:
        wjxx_0_[key_] += plus_n
    else:
        wjxx_1_[key_] += plus_n


def pjxx_xh_1_0(no, tjxx_, pjxx_crcs):
    # while not stop_thread:
    wjxx.update({'zkpj': [None, None], 'qbpj': [[], []], 'zjpj': None})
    strStartDateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    startSeconds = time.mktime(time.strptime(strStartDateTime, "%Y-%m-%d %X"))
    if pjxx_crcs is not None:
        pjxx = copy.deepcopy(pjxx_crcs)
    else:
        pjxx = pjxx_csh()
    # pjxx = {'zpm': [12, 3, 12, 7, 3, 5, 3, 4, 7, 11, 10, 11, 1, 1, 13, 4, 6, 3, 6, 13, 10, 12, 7, 12, 2, 2, 7, 11, 13, 9, 2, 8, 4, 1, 5, 4, 10, 10, 5, 5, 2, 11], 'wjxx': [{'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 1, 'pm': [1, 8, 8, 6, 9], 'pmsl': [1, 0, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 0], 'sysl': [3, 4, 4, 4, 4, 3, 4, 2, 3, 4, 4, 4, 3]}, {'dcpm': [7], 'dcfs': [49], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 1, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0, 'pm': [9, 8, 6, 9], 'pmsl': [0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0], 'sysl': [4, 4, 4, 4, 4, 3, 4, 3, 2, 4, 4, 4, 3]}], 'czgc': [{'wz': 0, 'cz': 7, 'pm': 13, 'bz': '补花', 'zjpm': [1, 8, 8, 6]}], 'loser_zfz': 0, 'fp': 4, 'zj': 0, 'syps': 42, 'fbh_syps': 39}
    pjxx_cs = copy.deepcopy(pjxx)
    fs_tt, pjxx_2 = 0, None
    dfwz = 1 - pjxx['zj']
    pjxx['wjxx'][dfwz]['dcpm'], pjxx['wjxx'][dfwz]['dcfs'], bool_rt = pmsl_get_dcp(pjxx['wjxx'][dfwz]['pmsl'], [])
    # print("牌局初始化完毕", no, pjxx['wjxx'][dfwz]['dcpm'], pjxx['wjxx'][dfwz]['dcfs'], bool_rt, pjxx)
    # print(dfwz, wjxx['fs'][dfwz], pjxx['wjxx'][dfwz]['dcpm'], pjxx['wjxx'][dfwz]['dcfs'], bool_rt)
    if bool_rt:
        if dfwz == 0:
            if getnp_bt(pjxx, dfwz, 0, 1):
                pjxx['wjxx'][dfwz]['btzt'] = 1
                add_czgc(pjxx, dfwz, 8, 0, '闲家天听')
            bool_rt = False
        else:
            pjxx_2 = copy.deepcopy(pjxx)
            pjxx_2['wjxx'][dfwz]['btzt'] = 1
            add_czgc(pjxx_2, dfwz, 8, 0, '闲家天听')
            pjxx_2 = xunhuancaozuo_1_0(pjxx_2, pjxx_2['zj'], 0)
            fs_tt = pjxx_2['jsfs'][1]
            # print("天听枚举完毕", fs_tt, pjxx_2)
    pjxx_3 = xunhuancaozuo_1_0(pjxx, pjxx['zj'], 0)
    fs = pjxx_3['jsfs'][1]
    # print("枚举完毕", fs, pjxx_3)
    if bool_rt:
        if fs_tt > fs:
            fs_best = fs_tt
            pjxx_best = pjxx_2
        elif fs_tt == fs:
            if pjxx_2['loser_zfz'] > pjxx_3['loser_zfz']:
                fs_best = fs_tt
                pjxx_best = pjxx_2
            else:
                fs_best = fs
                pjxx_best = pjxx_3
        else:
            fs_best = fs
            pjxx_best = pjxx_3
    else:
        fs_best = fs
        pjxx_best = pjxx_3
    strEndDateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    endSeconds = time.mktime(time.strptime(strEndDateTime, "%Y-%m-%d %X"))
    # threadLock.acquire()
    tjxx_['zjs'] += 1
    if fs_best < 0:
        tjxx_['win'] += 1
    tjxx_['score'] -= fs_best
    # mysql_insert(pjxx, pjxx_best)
    usetime = endSeconds - startSeconds
    tjxx_['time'] += usetime
    # threadLock.release()
    print(no, "最佳牌局", f'耗费时间：{usetime :.0f}秒', pjxx['zj'], fs_best, pjxx_best, wjxx['zjpj'],
          pjxx_best == wjxx['zjpj'], pjxx)
    if pjxx_best != wjxx['zjpj']:
        print(no, "pjxx_best != wjxx['zjpj']", pjxx_cs, pjxx_best, wjxx['zjpj'])
        tjxx_['error_1'] += 1
    # mysql_insert_pjxx(pjxx, pjxx_best, 'moni_221202')
    # pjxx_best2 = pjxx_xh(no, tjxx_, pjxx_cs)
    # # print(pjxx_best['jsfs'][1], pjxx_best2['jsfs'][1])
    # if pjxx_best != pjxx_best2:
    #     print(no, "pjxx_best != pjxx_best2", pjxx_best['jsfs'][1], pjxx_best2['jsfs'][1], pjxx_cs, pjxx_best, pjxx_best2)
    #     tjxx_['error_2'] += 1


def poolrrror(value):
    print("poolError", value)


def mysql_insert_pjxx(pjxx_cs, pjxx_end_, sjk):
    # db = pymysql.connect(host='rm-bp13rkrhg88b0lcq54o.mysql.rds.aliyuncs.com',
    #                      user='erdou_scf2021', password='Yu20521@#%$',
    #                      database='jisuerma', charset='utf8')
    db = pymysql.connect(host='127.0.0.1',
                         user='root', password='Hxm19910801Yzs',
                         database='jisuerma',
                         charset='utf8')
    cursor = db.cursor()
    sql = "insert into " + sjk + "(pjxx_cs,pjxx_end) VALUES (\"%s\", \"%s\")" % (pjxx_cs, pjxx_end_)
    # noinspection PyBroadException
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print("insert wrong", sql)
        db.rollback()
    db.close


def mysql_insert_pjxxsz(pjxx_cs, pjxx_end_y, sjk):
    for pjxx_end_ in pjxx_end_y:
        db = pymysql.connect(host='rm-bp13rkrhg88b0lcq54o.mysql.rds.aliyuncs.com',
                             user='erdou_scf2021', password='Yu20521@#%$',
                             database='jisuerma', charset='utf8')
        cursor = db.cursor()
        sql = "insert into " + sjk + "(pjxx_cs,pjxx_end) VALUES (\"%s\", \"%s\")" % (pjxx_cs, pjxx_end_)
        # noinspection PyBroadException
        try:
            cursor.execute(sql)
            db.commit()
        except:
            print("insert wrong", sql)
            db.rollback()
        db.close


if __name__ == '__main__':
    # 枚举所有成牌牌型，以便判断是否成牌或待成牌，共计216种
    for px in all_pxxx:
        print(px)
    print(len(all_pxxx))
    '''
    pjxx__ = {'zpm': [3, 11, 8, 10, 2, 11, 13, 1, 6, 12, 11, 12, 6, 13, 11, 6, 12, 10, 3, 2, 13, 13, 7, 8,
                      1, 4, 12, 9, 2, 1, 10, 4, 7, 3, 2, 5, 4, 10, 9, 8, 7, 9, 3],
              'wjxx': [{'dcpm': [4, 7], 'dcfs': [36, 39], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0,
                        'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq':[], 'ycp_th':[],
                        'ycp_tq_df':[], 'ycp_th_df':[], 'bhs': 0, 'pm': [6, 5, 7, 4],
                        'pmsl': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       'sysl': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]},
                       {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0,
                        'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq':[], 'ycp_th':[], 'ycp_tq_df':[],
                        'ycp_th_df':[], 'bhs': 0, 'pm': [8, 5, 9, 5, 1],
                        'pmsl': [1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0],
                        'sysl': [3, 4, 4, 4, 2, 4, 4, 3, 3, 4, 4, 4, 4]}],
              'czgc': [], 'loser_zfz': 0, 'fp': 9, 'zj': 1, 'syps': 43, 'fbh_syps': 39}
    getnp_cp(pjxx__, 1)
    bt = getnp_bt(pjxx__, 0, 0, 1)
    print(bt)
    pm_cp = 1
    pjxx__['wjxx'][1]['pmsl'][pm_cp - 1] -= 1
    pmsz_del(pjxx__['wjxx'][1]['pm'], [pm_cp])
    bt = getnp_bt(pjxx__, 1, 1, 2)
    print(bt)
    pjxx__ = {'zpm': [3, 11, 8, 10, 2, 11, 13, 1, 6, 12, 11, 12, 6, 13, 11, 6, 12, 10, 3, 2, 13, 13, 7, 8,
                      1, 4, 12, 9, 2, 1, 10, 4, 7, 3, 2, 5, 4, 10, 9, 8, 7, 9, 3],
              'wjxx': [{'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0,
                        'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq':[], 'ycp_th':[],
                        'ycp_tq_df':[], 'ycp_th_df':[1], 'bhs': 0, 'pm': [10, 10, 7, 4],
                        'pmsl': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       'sysl': [4, 4, 4, 3, 4, 4, 3, 4, 4, 4, 1, 4, 4]},
                       {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0,
                        'mpq_ag': 0, 'btzt': 1, 'ycp': [], 'ycp_tq':[], 'ycp_th':[10], 'ycp_tq_df':[10],
                        'ycp_th_df':[], 'bhs': 0, 'pm': [8, 5, 9, 5],
                        'pmsl': [1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0],
                        'sysl': [3, 4, 4, 4, 2, 4, 4, 3, 3, 4, 4, 4, 4]}],
              'czgc': [], 'loser_zfz': 0, 'fp': 9, 'zj': 1, 'syps': 43, 'fbh_syps': 39}
    p = getnp_p(pjxx__, 0, 1)
    print(p)
    '''
    # 多线程
    manager = Manager()
    tjxx = manager.dict()
    tjxx.update({'zjs': 0, 'lj': 0, 'time': 0, 'error_1': 0, 'error_2': 0})
    wjxx_0 = manager.dict()
    wjxx_0.update({'th': 0, 'dh': 0, 'xj_tt': 0, 'zj_tt': 0, 'bt': 0, 'p': 0, 'mg': 0, 'ag': 0,
                   'gskh': 0, 'bhs': 0, 'dp': 0, 'zm': 0, 'yj': 0, 'zj': 0, 'fs': 0})
    wjxx_1 = manager.dict()
    wjxx_1.update({'th': 0, 'dh': 0, 'xj_tt': 0, 'zj_tt': 0, 'bt': 0, 'p': 0, 'mg': 0, 'ag': 0,
                   'gskh': 0, 'bhs': 0, 'dp': 0, 'zm': 0, 'yj': 0, 'zj': 0, 'fs': 0})
    # 需要遍历的牌局总数
    n_pj = 50000
    # pjxx_all = []
    # test for pjxx_match
    # for i in range(n_pj):
    #     pjxx_ls = pjxx_csh()
    #     pjxx_all.append(pjxx_ls)
    # print(len(pjxx_all))
    # np.save('.//pjxx_all', pjxx_all)
    # pjxx_all = np.load('.//pjxx_all.npy', allow_pickle=True)
    # print(len(pjxx_all))
    # pjxx_ls = {'zpm': [13, 3, 2, 13, 11, 5, 2, 4, 6, 1, 9, 8, 3, 5, 7, 1, 2, 7, 4, 12, 7, 10, 9, 8, 12, 5, 6, 13, 3, 6, 13, 9, 12, 1, 4, 9, 6, 5, 7, 10, 8, 4, 10], 'wjxx': [{'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0, 'pm': [1, 11, 8, 11], 'pmsl': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0], 'sysl': [3, 4, 4, 4, 4, 4, 4, 3, 4, 4, 2, 4, 4]}, {'dcpm': [], 'dcfs': [], 'win_syps': [], 'mpq_pmsz': [], 'mpq_p': 0, 'mpq_mg': 0, 'mpq_ag': 0, 'btzt': 0, 'ycp': [], 'ycp_tq': [], 'ycp_th': [], 'ycp_tq_df': [], 'ycp_th_df': [], 'bhs': 0, 'pm': [2, 3, 12, 11, 10], 'pmsl': [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], 'sysl': [4, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4]}], 'czgc': [], 'loser_zfz': 0, 'fp': 6, 'zj': 1, 'syps': 43, 'fbh_syps': 39}
    # pjxx_xh(0, tjxx, pjxx_ls)
    # pjxx_xh_1_0(0, tjxx, pjxx_ls)
    # for i in range(n_pj):
    #     print(i)
    #     # pjxx_xh(i, tjxx, pjxx_all[i])
    #     pjxx_xh_1_0(i, tjxx, pjxx_all[i])
    # print(tjxx)
    # '''
    # pjxx_xh(0, tjxx)
    # '''
    # 线程数
    pool = Pool(processes=1)
    for i_ in range(n_pj):
        # pool.apply_async(pjxx_xh, (i_, tjxx, None, wjxx_0, wjxx_1, ), error_callback=poolrrror)
        pjxx_xh(i_, tjxx, None, wjxx_0, wjxx_1)
        # pool.apply_async(pjxx_xh_1_0, (i_, tjxx, pjxx_all[i_], ), error_callback=poolrrror)
    pool.close()
    pool.join()
    for key in wjxx_0.keys():
        if key not in ['fs', 'bhs']:
            wjxx_0[key] = round(wjxx_0[key] / tjxx['zjs'] * 100, 2)
    for key in wjxx_1.keys():
        if key not in ['fs', 'bhs']:
            wjxx_1[key] = round(wjxx_1[key] / tjxx['zjs'] * 100, 2)
    print(tjxx)
    print(wjxx_0)
    print(wjxx_1)

