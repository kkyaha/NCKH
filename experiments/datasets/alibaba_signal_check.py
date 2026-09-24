# -*- coding: utf-8 -*-
"""
Kiem tra Alibaba v2021 co chua tin hieu workload -> resource khong
===================================================================
Chay DUNG phep kiem da phoi ra van de cua RCAEval, de so sanh truc tiep.

Ket qua tren RCAEval (Sock Shop, gop toan bo 90 run):
    R^2 trung vi: CPU 0.0034 | Memory 0.0034 | Socket 0.0038
    bien do tai tuong doi (max-min)/mean = 1.70
=> workload giai thich duoi 0.4% phuong sai tai nguyen. Khong mo hinh nao
   trich duoc quan he khong ton tai, nen moi so sanh do chinh xac du bao tren
   benchmark do la do tieng on.

Gia thuyet o day: Alibaba v2021 la trace PRODUCTION 12 gio, tai bien thien
that (khong phai load test o trang thai dung nhu benchmark RCA), nen bien do
tai phai rong hon va tin hieu phai xuat hien.

Nguon du lieu (moi bang tai 1 chunk de lay mau):
  MS_Resource_Table : timestamp, msname, msinstanceid, nodeid,
                      cpu_utilization, memory_utilization
  MS_MCR_RT_Table   : timestamp, msname, msinstanceid, metrics, value
                      metrics gom *_MCR (call rate = WORKLOAD) va *_RT (latency)

Ghep theo (timestamp, msname) hoac (timestamp, msinstanceid) neu co.

Output: data/processed/scm_results/alibaba_signal_check.csv
"""

import os
import sys
import glob
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
# Mac dinh tro toi ban sao CO DINH ngoai repo (45GB, khong the commit).
# Doi bang bien moi truong ALIBABA_DIR neu de cho khac.
DATA_DIR = os.environ.get('ALIBABA_DIR',
                          os.path.expanduser('~/NCKH-data/alibaba'))

MIN_POINTS = 60          # so diem toi thieu moi service de fit
MCR_METRICS = ('providerRPC_MCR', 'consumerRPC_MCR', 'HTTP_MCR', 'consumerMQ_MCR')
# Ten cot THUC TE trong file (khac tai lieu): cot dau la index khong ten,
# resource dung instance_cpu_usage / instance_memory_usage, va bang MCR dung
# 'metric' (so it) chu khong phai 'metrics'.
RES_COLS = ['msname', 'instance_cpu_usage', 'instance_memory_usage', 'timestamp']
MCR_COLS = ['timestamp', 'msname', 'metric', 'value']
CHUNK = 2_000_000


def _find(pattern):
    hits = glob.glob(os.path.join(DATA_DIR, '**', pattern), recursive=True)
    return sorted(hits)


def load_resource():
    """Doc theo chunk, chi giu cot can, ha kieu de vua RAM 16GB."""
    files = [f for f in _find('*.csv') if 'Resource' in os.path.basename(f)]
    if not files:
        return None
    parts = []
    for f in files:
        for ch in pd.read_csv(f, usecols=RES_COLS, chunksize=CHUNK,
                              dtype={'msname': 'category',
                                     'instance_cpu_usage': 'float32',
                                     'instance_memory_usage': 'float32',
                                     'timestamp': 'int64'}):
            parts.append(ch.groupby(['timestamp', 'msname'], observed=True)
                         .agg(cpu=('instance_cpu_usage', 'mean'),
                              mem=('instance_memory_usage', 'mean'))
                         .reset_index())
    df = pd.concat(parts, ignore_index=True)
    # gop lai vi mot microservice co the trai qua nhieu chunk
    return (df.groupby(['timestamp', 'msname'], observed=True)
            .agg(cpu=('cpu', 'mean'), mem=('mem', 'mean')).reset_index())


def load_mcr():
    """Workload = tong cac metric *_MCR cua cung (timestamp, msname).

    LUU Y: cac chunk tai ve co the KHONG lien tuc ve thoi gian (vi du co chunk
    0,1,3 nhung thieu 2). Dieu do khong sao cho phep kiem nay: day la hoi quy
    cat ngang workload -> resource tren cac cap (timestamp, service), khong
    phai mo hinh chuoi thoi gian, nen mot khoang trong chi lam giam so diem
    chu khong lam lech uoc luong."""
    files = [f for f in _find('*.csv')
             if any(k in os.path.basename(f) for k in ('RTQps', 'Qps'))]
    if not files:
        return None
    parts = []
    for f in files:
        for ch in pd.read_csv(f, usecols=MCR_COLS, chunksize=CHUNK,
                              dtype={'msname': 'category', 'metric': 'category',
                                     'value': 'float32', 'timestamp': 'int64'}):
            ch = ch[ch['metric'].isin(MCR_METRICS)]
            if len(ch):
                parts.append(ch.groupby(['timestamp', 'msname'], observed=True)['value']
                             .sum().reset_index())
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return (df.groupby(['timestamp', 'msname'], observed=True)['value']
            .sum().reset_index().rename(columns={'value': 'workload'}))


def r2_of(X, y):
    X = np.asarray(X, float).reshape(-1, 1)
    y = np.asarray(y, float)
    if len(y) < MIN_POINTS or np.std(X) < 1e-12 or np.std(y) < 1e-12:
        return None
    m = LinearRegression().fit(X, y)
    return float(1 - np.sum((y - m.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)), \
        float(m.coef_[0])


def main():
    print("=== loading (chunked) ===")
    rs, wl = load_resource(), load_mcr()
    if rs is None or wl is None:
        print("  khong tim thay file CSV trong", DATA_DIR)
        return
    print(f"  resource: {len(rs):,} dong, {rs.msname.nunique():,} microservice")
    print(f"  workload: {len(wl):,} dong, {wl.msname.nunique():,} microservice")
    print(f"  ts resource: {rs.timestamp.min()} .. {rs.timestamp.max()}")
    print(f"  ts workload: {wl.timestamp.min()} .. {wl.timestamp.max()}")

    df = wl.merge(rs, on=['timestamp', 'msname'], how='inner')
    print(f"\nghep duoc: {len(df):,} dong | {df.msname.nunique():,} microservice")
    if df.empty:
        print("  ghep RONG -- hai chunk khong trung khoang thoi gian/service")
        return

    rows = []
    for name, g in df.groupby('msname', observed=True):
        g = g[g.workload > 0]
        if len(g) < MIN_POINTS:
            continue
        rel = float((g.workload.max() - g.workload.min()) / g.workload.mean())
        for metric in ('cpu', 'mem'):
            if metric not in g.columns:
                continue
            out = r2_of(g.workload.values, g[metric].values)
            if out is None:
                continue
            r2, slope = out
            rows.append(dict(msname=str(name), metric=metric, n=len(g),
                             rel_range=rel, r2=r2, slope=slope))

    d = pd.DataFrame(rows)
    if d.empty:
        print("  khong du service dat nguong MIN_POINTS")
        return
    d.to_csv(os.path.join(OUT_DIR, 'alibaba_signal_check.csv'), index=False)

    print(f"\n{'='*70}")
    print("  ALIBABA v2021: workload -> resource")
    print(f"{'='*70}")
    g = d.groupby('metric').agg(
        n_services=('r2', 'size'), R2_median=('r2', 'median'),
        R2_p75=('r2', lambda x: x.quantile(0.75)),
        R2_max=('r2', 'max'),
        pct_R2_gt_0_3=('r2', lambda x: 100 * (x > 0.3).mean()),
        rel_range_median=('rel_range', 'median')).round(4)
    print(g.to_string())

    print(f"\n{'='*70}")
    print("  DOI CHIEU voi RCAEval (Sock Shop)")
    print(f"{'='*70}")
    print(f"  {'':22s} {'RCAEval':>12s} {'Alibaba':>12s}")
    for metric, rcae in (('cpu', 0.0034), ('mem', 0.0034)):
        if metric in g.index:
            print(f"  {'R2 trung vi ' + metric:22s} "
                  f"{rcae:12.4f} {g.loc[metric, 'R2_median']:12.4f}")
    print(f"  {'bien do tai':22s} {1.70:12.2f} "
          f"{d.rel_range.median():12.2f}")
    print("\n  Ket luan: Alibaba dung duoc cho nua forecasting NEU R2 trung vi")
    print("  cao hon RCAEval mot bac ro rang (>= 0.1) hoac >=25% service co R2>0.3.")


if __name__ == '__main__':
    main()
