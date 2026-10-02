# -*- coding: utf-8 -*-
"""RQ6 — CHON TANG KHONG GIAM SAT: bien phat hien thanh PHUONG PHAP.

CHO HO CUA BAI. Cac thuc nghiem truoc (E1, E5b) tra loi: tang nao DUNG cho loai
loi nao. Nhung luc van hanh KHONG AI BIET TRUOC do la loi gi -- neu phai biet
loai loi de chon tang, thi phat hien chi la CHAN DOAN HAU NGHIEM, khong dung
duoc. Day la phan bien "dau la thu toi dung duoc".

CAU HOI: co chon duoc tang chi tu DU LIEU, khong nhan, khong biet loai loi?

Y tuong. Tang DUNG la tang ma nguyen nhan NOI BAT TREN NEN -- mot service nhoi
len han so voi cac service khac. Tang SAI la tang ma moi service nhieu deu nhau.
Do la mot tinh chat cua PHAN BO diem, do duoc ma khong can dap an. Bon thong ke
do "do nhoi", tat ca KHONG DUNG NHAN:

  P_tyle   = r_top1 / trung_vi(r)          -- dan dau noi bat hon muc tieu bieu bao nhieu
  P_khe    = (r_top1 - r_top2) / r_top1    -- khe giua nhat va nhi
  P_tutap  = r_top1 / sum(r)               -- muc do tap trung
  P_tho    = r_top1                        -- DO LON THO (doi chung am: thang do cac
                                              tang lech nhau hang chuc lan nen cai nay
                                              phai luon chon tang latency -> se that bai)

So voi:
  - luon dung MOT tang co dinh (cpu / latency / workload)
  - tron MOI cot metric (cach pho bien trong van lieu)
  - ORACLE: tang tot nhat cho tung ca (tran tren, khong dung duoc that)

Neu mot quy tac chon tang dat GAN oracle va VUOT moi tang co dinh, thi nguyen ly
khop tang tro thanh mot phuong phap van hanh duoc, khong chi mot phat hien.

Chay:  python papers/p2_khop_tang/experiments/rq6_layer_selection.py
Ra:    data/processed/scm_results/rq6_layer_selection.csv          (1 dong / run / tang)
       data/processed/scm_results/rq6_layer_selection_summary.csv  (1 dong / quy tac / nhom)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
RES = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')

LAYERS = ['workload', 'cpu', 'latency-50']
RESOURCE = ['cpu', 'mem', 'disk', 'socket']
ENV = {'time', 'imte', 'vm_cpu_util', 'vm_mem_avail_mb'}


def dich_chuyen(d, t, layer):
    """|dich chuyen trung binh| / std truoc, cho moi service tren MOT tang."""
    tc = 'imte' if 'imte' in d.columns else 'time'
    b, a = d[d[tc] < t], d[d[tc] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    out = {}
    for c in d.columns:
        if not c.endswith('_' + layer):
            continue
        s = c[:-(len(layer) + 1)]
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        out[s] = abs(na.mean() - nb.mean()) / nb.std()
    return out


def moi_cot(d, t):
    """Xep hang tren MOI cot metric, gop ve service (cach pho bien trong van lieu)."""
    tc = 'imte' if 'imte' in d.columns else 'time'
    b, a = d[d[tc] < t], d[d[tc] >= t]
    if len(b) < 20 or len(a) < 20:
        return {}
    diem = {}
    for c in d.columns:
        if c in ENV or '_' not in c:
            continue
        nb, na = b[c].dropna(), a[c].dropna()
        if len(nb) < 10 or len(na) < 10 or nb.std() == 0:
            continue
        s = c.rsplit('_', 1)[0]
        diem[s] = max(diem.get(s, 0.0), abs(na.mean() - nb.mean()) / nb.std())
    return diem


def cac_run():
    ss = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-SS')
    if os.path.isdir(ss):
        for scen in sorted(os.listdir(ss)):
            sp = os.path.join(ss, scen)
            if not os.path.isdir(sp):
                continue
            inj, _, ft = scen.rpartition('_')
            for r in sorted(os.listdir(sp)):
                mp, ip = os.path.join(sp, r, 'simple_metrics.csv'), os.path.join(sp, r, 'inject_time.txt')
                if os.path.exists(mp) and os.path.exists(ip):
                    try:
                        yield 'SockShop', inj, ft, r, pd.read_csv(mp), int(open(ip).read().strip())
                    except Exception:
                        continue
    for pref, root, name in (('re2ob_', 'RE2-OB', 'OnlineBoutique'),
                             ('re2tt_', 'trainticket', 'TrainTicket')):
        for sp in sorted(glob.glob(os.path.join(BASE_DIR, 'data', 'raw', root, pref + '*'))):
            base = os.path.basename(sp)[len(pref):]
            run = base.rsplit('_', 1)[-1]
            inj, _, ft = base.rsplit('_', 1)[0].rpartition('_')
            mp, ip = os.path.join(sp, 'metrics.parquet'), os.path.join(sp, 'inject_time.txt')
            if os.path.exists(mp) and os.path.exists(ip):
                try:
                    yield name, inj, ft, run, pd.read_parquet(mp), int(open(ip).read().strip())
                except Exception:
                    continue


def do_nhoi(v: dict):
    """Bon thong ke 'do nhoi' cua mot phan bo diem -- KHONG dung nhan."""
    if len(v) < 3:
        return None
    x = np.sort(np.array(list(v.values()), dtype=float))[::-1]
    top1, top2 = float(x[0]), float(x[1])
    med, tong = float(np.median(x)), float(x.sum())
    return dict(
        r_top1=top1, r_top2=top2, r_med=med, n=len(x),
        P_tyle=top1 / med if med > 0 else np.inf,
        P_khe=(top1 - top2) / top1 if top1 > 0 else 0.0,
        P_tutap=top1 / tong if tong > 0 else 0.0,
        P_tho=top1,
    )


QUY_TAC = ['P_tyle', 'P_khe', 'P_tutap', 'P_tho']


THU_TU_TANG = ['workload', 'cpu', 'latency-50']   # thuong nguon -> ha nguon


def quy_tac_thuong_nguon(g: pd.DataFrame, cot_P: str, tau: float):
    """QUY TAC SUY TU LY THUYET, khong phai heuristic.

    Menh de 1 (dang ba muc) noi: tang CHUA nguyen nhan co tin hieu manh nhat, va
    moi tang HAU DUE cua no CUNG co tin hieu (loang hon). Nen khi NHIEU tang cung
    phan ung, nguyen nhan nam o tang THUONG NGUON nhat trong so do -- cac tang
    con lai chi THUA HUONG tin hieu.

    Day chinh la cho quy tac "chon tang nhoi nhat" that bai: voi loi tai nguyen,
    tang latency la HAU DUE cua tang cpu nen cung nhoi, va doi khi nhoi hon.

    Quy tac: goi la "phan ung" moi tang co P >= tau * max(P). Trong so cac tang
    phan ung, chon tang THUONG NGUON nhat theo THU_TU_TANG.
    """
    gg = g.dropna(subset=[cot_P])
    if gg.empty:
        return None
    pmax = gg[cot_P].max()
    if not np.isfinite(pmax) or pmax <= 0:
        return None
    pu = gg[gg[cot_P] >= tau * pmax]
    for la in THU_TU_TANG:
        if (pu.layer == la).any():
            return la
    return gg.loc[gg[cot_P].idxmax(), 'layer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tu-cache', action='store_true',
                    help='doc lai rq6_layer_selection.csv thay vi quet du lieu tho')
    ap.add_argument('--tau', type=float, default=0.5)
    a = ap.parse_args()
    if a.tu_cache:
        D = pd.read_csv(os.path.join(RES, 'rq6_layer_selection.csv'))
        return phan_tich(D, a.tau)
    rows = []
    for he, inj, ft, run, d, t in cac_run():
        mc = moi_cot(d, t)
        hang_mc = np.nan
        if mc and inj in mc:
            hang_mc = sorted(mc, key=mc.get, reverse=True).index(inj) + 1
        for layer in LAYERS:
            v = dich_chuyen(d, t, layer)
            if len(v) < 3 or inj not in v:
                continue
            s = do_nhoi(v)
            if s is None:
                continue
            hang = sorted(v, key=v.get, reverse=True).index(inj) + 1
            rows.append(dict(he=he, injected=inj, fault=ft, run=run, layer=layer,
                             hang=hang, n_cand=len(v), hang_moi_cot=hang_mc, **s))
    D = pd.DataFrame(rows)
    D['nhom'] = np.where(D.fault.isin(RESOURCE), 'tai nguyen', 'mang')
    D.to_csv(os.path.join(RES, 'rq6_layer_selection.csv'), index=False)
    return phan_tich(D, a.tau)


def phan_tich(D, tau):
    print(f'{len(D)} dong = (run x tang) | {D.groupby(["he","injected","fault","run"]).ngroups} run\n')

    # ---- gop ve 1 dong / run, roi ap tung quy tac ----
    kl = ['he', 'injected', 'fault', 'run', 'nhom']
    ket = []
    for key, g in D.groupby(kl):
        r = dict(zip(kl, key))
        r['n_cand'] = g.n_cand.median()
        for la in LAYERS:
            gg = g[g.layer == la]
            r[f'hang_{la}'] = gg.hang.iloc[0] if len(gg) else np.nan
        r['hang_moi_cot'] = g.hang_moi_cot.iloc[0]
        for qt in QUY_TAC:
            gg = g.dropna(subset=[qt])
            if gg.empty:
                r[f'chon_{qt}'] = None
                r[f'hang_{qt}'] = np.nan
                continue
            la = gg.loc[gg[qt].idxmax(), 'layer']
            r[f'chon_{qt}'] = la
            r[f'hang_{qt}'] = gg[gg.layer == la].hang.iloc[0]
        # QUY TAC THUONG NGUON (suy tu ly thuyet), cho moi thong ke do nhoi
        for qt in QUY_TAC:
            la = quy_tac_thuong_nguon(g, qt, tau)
            r[f'chon_TN_{qt}'] = la
            r[f'hang_TN_{qt}'] = g[g.layer == la].hang.iloc[0] if la else np.nan
        r['hang_oracle'] = g.hang.min()
        r['layer_oracle'] = g.loc[g.hang.idxmin(), 'layer']
        ket.append(r)
    K = pd.DataFrame(ket)

    def ac(col, k=1):
        return 100 * (K[col] <= k).mean()

    print('=' * 104)
    print('  (1) AC@1 / AC@3 / MRR — moi chien luoc chon tang, TOAN BO 270 ca')
    print('=' * 104)
    chien_luoc = ([(f'tang co dinh: {la}', f'hang_{la}') for la in LAYERS]
                  + [('tron MOI cot metric', 'hang_moi_cot')]
                  + [(f'CHON nhoi-nhat: {qt}', f'hang_{qt}') for qt in QUY_TAC]
                  + [(f'CHON THUONG-NGUON: {qt}', f'hang_TN_{qt}') for qt in QUY_TAC]
                  + [('ORACLE (tran tren)', 'hang_oracle')])
    bang = []
    for ten, col in chien_luoc:
        v = K[col].dropna()
        bang.append(dict(chien_luoc=ten, n=len(v),
                         **{'AC@1': 100 * (v <= 1).mean(), 'AC@3': 100 * (v <= 3).mean(),
                            'AC@5': 100 * (v <= 5).mean(), 'MRR': (1 / v).mean()}))
    B = pd.DataFrame(bang)
    with pd.option_context('display.width', 200, 'display.float_format', lambda x: f'{x:,.3f}'):
        print(B.to_string(index=False))

    print('\n' + '=' * 104)
    print('  (2) THEO NHOM LOI — quy tac co chon DUNG tang khong?')
    print('=' * 104)
    for qt in [f'{q}' for q in QUY_TAC] + [f'TN_{q}' for q in QUY_TAC]:
        print(f'\n  --- {qt} ---')
        t = pd.crosstab(K.nhom, K[f'chon_{qt}'], normalize='index') * 100
        print(t.round(1).to_string())
        for nh, g in K.groupby('nhom'):
            v = g[f'hang_{qt}'].dropna()
            o = g['hang_oracle'].dropna()
            print(f'     {nh:11s} AC@1 = {100*(v<=1).mean():5.1f}%   (oracle {100*(o<=1).mean():5.1f}%)')

    print('\n' + '=' * 104)
    print('  (3) QUY TAC TOT NHAT vs TANG CO DINH TOT NHAT — kiem nhi thuc ghep cap')
    print('=' * 104)
    moi_qt = [f'hang_{q}' for q in QUY_TAC] + [f'hang_TN_{q}' for q in QUY_TAC]
    tot_col = max(moi_qt, key=lambda c: (K[c] <= 1).mean())
    tot_qt = tot_col[len('hang_'):]
    tot_cd = max(LAYERS + ['moi_cot'], key=lambda l: (K[f'hang_{l}'] <= 1).mean())
    a = (K[f'hang_{tot_qt}'] <= 1).astype(float)
    b = (K[f'hang_{tot_cd}'] <= 1).astype(float)
    hon, kem = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(hon, hon + kem, 0.5).pvalue if hon + kem else np.nan
    print(f'  quy tac tot nhat   : CHON theo {tot_qt}      AC@1 = {100*a.mean():.1f}%')
    print(f'  tang co dinh tot nhat: {tot_cd:12s}        AC@1 = {100*b.mean():.1f}%')
    print(f'  ca quy tac DUNG / co dinh SAI : {hon}')
    print(f'  ca quy tac SAI / co dinh DUNG : {kem}')
    print(f'  McNemar (nhi thuc ghep cap)   : p = {p:.4g}')

    print('\n' + '=' * 104)
    print(f'  (3b) QUET tau cho quy tac THUONG NGUON (tau hien tai = {tau})')
    print('=' * 104)
    for t_ in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ds = []
        for key, g in D.groupby(['he', 'injected', 'fault', 'run']):
            la = quy_tac_thuong_nguon(g, 'P_tutap', t_)
            if la is None:
                continue
            ds.append(g[g.layer == la].hang.iloc[0])
        ds = pd.Series(ds)
        print(f'    tau = {t_:4.2f}   AC@1 = {100*(ds<=1).mean():5.1f}%   '
              f'AC@3 = {100*(ds<=3).mean():5.1f}%   MRR = {(1/ds).mean():.3f}')

    print('\n' + '=' * 104)
    print('  (4) THEO HE — quy tac co tong quat hoa qua ba he khong?')
    print('=' * 104)
    for he, g in K.groupby('he'):
        d = {f'{la}': 100 * (g[f'hang_{la}'] <= 1).mean() for la in LAYERS}
        d['moi_cot'] = 100 * (g.hang_moi_cot <= 1).mean()
        d[f'CHON {tot_qt}'] = 100 * (g[f'hang_{tot_qt}'] <= 1).mean()
        d['TN P_tutap'] = 100 * (g['hang_TN_P_tutap'] <= 1).mean()
        d['ORACLE'] = 100 * (g.hang_oracle <= 1).mean()
        print(f'  {he:15s} ' + '  '.join(f'{k}={v:5.1f}%' for k, v in d.items()))

    B.to_csv(os.path.join(RES, 'rq6_layer_selection_summary.csv'), index=False)
    K.to_csv(os.path.join(RES, 'rq6_layer_selection_per_run.csv'), index=False)
    print(f'\n-> {RES}/rq6_layer_selection.csv')
    print(f'-> {RES}/rq6_layer_selection_per_run.csv')
    print(f'-> {RES}/rq6_layer_selection_summary.csv')


if __name__ == '__main__':
    main()
