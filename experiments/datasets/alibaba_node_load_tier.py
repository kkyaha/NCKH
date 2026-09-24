# -*- coding: utf-8 -*-
"""
Tang 1.5: tai node co cai thien du bao resource khong?
=======================================================
Dong co
-------
alibaba_confounder_test.py cho thay ton tai phu thuoc giua cac instance DONG
NODE ma do thi hai tang khong mo hinh hoa: tuong quan rieng phan (sau khi khu
workload ca hai) la 0.398 cho cap dong-node vs 0.104 cho cap khac-node,
common-language effect 0.873 tren 4000 cap moi nhom.

Diem then chot: TAI NODE LA DAI LUONG SUY RA DUOC, khong phai an so. Tai cua mot
node = tong workload cua cac instance dat tren no, va tat ca chung deu nam duoi
gateway. Nen khi can thiep do(gateway), tai node cam sinh TINH DUOC -- khac han
cac covariate da that bai truoc day (memory, socket), von phai ngoai suy rieng
nen gop them phuong sai thay vi thong tin.

Nho vay co the giu nguyen can thiep tai gateway (Proposition 1 khong phai sua)
ma van dung duoc confounder da do.

So sanh
-------
  constant   : du doan trung binh tap train (moc skill)
  base       : cpu_i ~ workload_i                    (mo hinh hai tang hien tai)
  +node      : cpu_i ~ workload_i + node_load        (mo hinh ba tang)

Chot an toan giu nguyen: baseline hang so, negative control (hoan vi CA HAI bien
du bao trong tung instance), tach in-distribution / OOD, skill thay MAPE.

LUU Y VE PHAM VI: script dung tai node QUAN SAT DUOC. Do la can tren cua loi ich
-- luc du bao that, tai node phai duoc tinh tu cac workload da lan truyen qua
Tier-1, nen sai so cua Tier-1 se cong vao. Neu phep kiem nay cho ket qua am thi
khong can lam buoc kia; neu duong thi phai danh gia lai voi tai node DU DOAN.

Output: data/processed/scm_results/alibaba_node_load_tier.csv
"""

import os
import sys
import glob
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(__file__))
from alibaba_signal_check import DATA_DIR, MCR_METRICS   # noqa: E402

CHUNK = 2_000_000
MIN_POINTS = 80
MAX_INSTANCES = 3000
N_FIT = 2000
SEED = 42
RNG = np.random.RandomState(SEED)


def load_joined():
    """(timestamp, instance) -> cpu, workload, nodeid ; kem tai node cong don."""
    res = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'MSResource_*.csv'))):
        for ch in pd.read_csv(f, usecols=['timestamp', 'msinstanceid', 'nodeid',
                                          'instance_cpu_usage'],
                              chunksize=CHUNK,
                              dtype={'msinstanceid': 'str', 'nodeid': 'str',
                                     'instance_cpu_usage': 'float32',
                                     'timestamp': 'int64'}):
            res.append(ch)
    res = pd.concat(res, ignore_index=True)

    wl = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'MSRTQps_*.csv'))):
        for ch in pd.read_csv(f, usecols=['timestamp', 'msinstanceid',
                                          'metric', 'value'],
                              chunksize=CHUNK,
                              dtype={'msinstanceid': 'str', 'metric': 'category',
                                     'value': 'float32', 'timestamp': 'int64'}):
            ch = ch[ch['metric'].isin(MCR_METRICS)]
            if len(ch):
                wl.append(ch.groupby(['timestamp', 'msinstanceid'],
                                     observed=True)['value'].sum().reset_index())
    wl = (pd.concat(wl, ignore_index=True)
          .groupby(['timestamp', 'msinstanceid'], observed=True)['value']
          .sum().reset_index().rename(columns={'value': 'workload'}))

    df = res.merge(wl, on=['timestamp', 'msinstanceid'], how='inner')
    # Tang 1.5: tai node = tong workload cac instance dong node, cung thoi diem
    nl = (df.groupby(['timestamp', 'nodeid'], observed=True)['workload']
          .sum().reset_index().rename(columns={'workload': 'node_load'}))
    df = df.merge(nl, on=['timestamp', 'nodeid'], how='left')
    # tru chinh minh -> tai cua CAC instance KHAC tren cung node
    df['node_load_others'] = df['node_load'] - df['workload']
    return df


def skill(y, pred, base):
    mm = float(np.mean((y - pred) ** 2)); mb = float(np.mean((y - base) ** 2))
    return np.nan if mb <= 1e-15 else 1.0 - mm / mb


def _split(g, mode):
    if mode == 'ood':
        g = g.sort_values('workload')
        k = int(len(g) * 0.67)
        return g.iloc[:k], g.iloc[k:]
    idx = RNG.permutation(len(g))
    k = int(len(g) * 0.7)
    return g.iloc[idx[:k]], g.iloc[idx[k:]]


def eval_instance(g, mode, shuffled):
    tr, te = _split(g, mode)
    if len(tr) < 40 or len(te) < 20:
        return []
    if len(tr) > N_FIT:
        tr = tr.sample(N_FIT, random_state=SEED)
    y_tr, y_te = tr.cpu.values, te.cpu.values
    if np.std(y_tr) < 1e-12:
        return []

    const = np.full(len(y_te), float(np.mean(y_tr)))
    out = [dict(mode=mode, shuffled=shuffled, model='constant', skill=0.0,
                n_train=len(tr), n_test=len(te))]

    specs = {'base': ['workload'], '+node': ['workload', 'node_load_others']}
    for name, cols in specs.items():
        Xtr, Xte = tr[cols].values, te[cols].values
        if np.any(np.std(Xtr, axis=0) < 1e-12):
            continue
        p = LinearRegression().fit(Xtr, y_tr).predict(Xte)
        out.append(dict(mode=mode, shuffled=shuffled, model=name,
                        skill=skill(y_te, p, const),
                        n_train=len(tr), n_test=len(te)))
    return out


def run():
    print("=== loading ===")
    df = load_joined()
    print(f"  {len(df):,} dong | {df.msinstanceid.nunique():,} instance | "
          f"{df.nodeid.nunique():,} node")

    cnt = df.groupby('msinstanceid').size()
    keep = cnt[cnt >= MIN_POINTS].index
    if len(keep) > MAX_INSTANCES:
        keep = RNG.choice(np.asarray(keep), MAX_INSTANCES, replace=False)
    df = df[df.msinstanceid.isin(set(keep))]
    df = df.rename(columns={'instance_cpu_usage': 'cpu'})
    df = df[(df.workload > 0) & np.isfinite(df.node_load_others)]
    print(f"  dung {df.msinstanceid.nunique():,} instance")

    rows = []
    for shuffled in (False, True):
        work = df
        if shuffled:
            work = df.copy()
            for c in ('workload', 'node_load_others'):
                work[c] = (work.groupby('msinstanceid', observed=True)[c]
                           .transform(lambda s: RNG.permutation(s.values)))
        for inst, g in work.groupby('msinstanceid', observed=True):
            if len(g) < MIN_POINTS:
                continue
            for mode in ('indist', 'ood'):
                for r in eval_instance(g, mode, shuffled):
                    r['msinstanceid'] = inst
                    rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'alibaba_node_load_tier.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu"); return
    d = d.replace([np.inf, -np.inf], np.nan)
    n_inst = d.msinstanceid.nunique()
    print(f"\n{'='*70}\n  TANG 1.5 (tai node)  --  {n_inst:,} instance\n{'='*70}")
    print(f"  {'model':10s} {'mode':7s} {'shuf':6s} {'skill_med':>10s} {'%>0':>7s}")
    for mode in ('indist', 'ood'):
        for shuf in (False, True):
            for m in ('constant', 'base', '+node'):
                x = d[(d['mode'] == mode) & (d.shuffled == shuf) & (d.model == m)]
                if x.empty:
                    continue
                print(f"  {m:10s} {mode:7s} {str(shuf):6s} "
                      f"{x.skill.median():10.4f} {100*(x.skill > 0).mean():6.1f}%")

    print(f"\n{'='*70}\n  Tai node co cai thien khong? (chi shuf=False)\n{'='*70}")
    for mode in ('indist', 'ood'):
        piv = (d[(d['mode'] == mode) & (~d.shuffled)]
               .pivot_table(index='msinstanceid', columns='model', values='skill')
               .dropna(subset=['base', '+node']))
        if piv.empty:
            continue
        diff = (piv['+node'] - piv['base']).dropna()
        st, p = stats.wilcoxon(diff)
        win = float((piv['+node'] > piv['base']).mean())
        med = float(np.median(diff))
        verdict = ('CAI THIEN dang ke' if p < 0.05 and med > 0.01 and win >= 0.6
                   else 'khong dang ke')
        print(f"  {mode:7s} n={len(piv):4d} | base={piv['base'].median():+.4f} "
              f"-> +node={piv['+node'].median():+.4f}")
        print(f"          median diff={med:+.4f}  p={p:.3g}  win={100*win:.1f}%"
              f"  -> {verdict}")


if __name__ == '__main__':
    report(run())
