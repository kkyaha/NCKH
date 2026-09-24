# -*- coding: utf-8 -*-
"""
Alibaba co confounder ha tang khong? -- phep kiem co the LAT ket luan ve do()
=============================================================================
Lap luan "toan tu do() rong" gom hai khau:

  (1) Can thiep tai node in-degree 0, DAG khong latent confounder
      => khong co backdoor path => P(Y|do X) = P(Y|X) DONG NHAT.
      Day la DINH LY, khong phu thuoc du lieu.

  (2) "Khong co latent confounder" -- khau nay DUOC DO, va do tren RCAEval:
      dong-node p=0.546, tai toan cuc giam tuong quan du 1.2%.

Khau (2) co the KHONG chuyen sang Alibaba, va co ly do manh de nghi ngo:

      | so service | container | tranh chap tai nguyen
  SS  |      7     |   ~13     | gan nhu khong
  Ali |   1,303    |  90,000   | rat nhieu

Voi 90k container tren ha tang dung chung, confounding do tranh chap node la
hoan toan hop ly. Neu co that thi do() CO noi dung tai day, va ket luan cua
chung ta phai doi thanh: "do() rong tren topology nho, co noi dung tren ha tang
dung chung quy mo lon".

Thiet ke
--------
Don vi phan tich la INSTANCE (msinstanceid), khong phai service: co-location la
tinh chat cua container. Mot service co nhieu instance tren nhieu node.

Voi moi cap instance (i, j):
  - tuong quan RIENG PHAN giua cpu_i va cpu_j SAU khi khu workload_i va
    workload_j (tuc khu het cac cha nhan qua da khai bao trong do thi hai tang)
  - so sanh nhom DONG-NODE voi nhom KHAC-NODE

Neu do thi hai tang la du, phan du phai gan doc lap o ca hai nhom. Neu nhom
dong-node co tuong quan du CAO HON dang ke -> ton tai duong phu thuoc chua duoc
mo hinh hoa -> ung vien cho mot node nguyen nhan chung -> do() co viec de lam.

Chi dung cap KHONG co canh call-graph, de tach confounding khoi duong nhan qua
that.

Output: data/processed/scm_results/alibaba_confounder_test.csv
"""

import os
import sys
import glob
import random
import warnings
import itertools

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
from alibaba_signal_check import DATA_DIR, MCR_METRICS   # noqa: E402
from alibaba_tier1_propagation import build_topology     # noqa: E402

CHUNK = 2_000_000
MIN_POINTS = 60          # so diem thoi gian toi thieu moi instance
MAX_INSTANCES = 1500     # lay mau instance de vua RAM
MAX_PAIRS = 4000         # so cap moi nhom
SEED = 42
RNG = np.random.RandomState(SEED)


def load_instance_resource():
    """(timestamp, msinstanceid) -> cpu, kem nodeid va msname."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'MSResource_*.csv')))
    parts = []
    for f in files:
        for ch in pd.read_csv(
                f, usecols=['timestamp', 'msname', 'msinstanceid', 'nodeid',
                            'instance_cpu_usage'],
                chunksize=CHUNK,
                dtype={'msname': 'str', 'msinstanceid': 'str', 'nodeid': 'str',
                       'instance_cpu_usage': 'float32', 'timestamp': 'int64'}):
            parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def load_instance_workload():
    """(timestamp, msinstanceid) -> workload (tong cac metric *_MCR)."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'MSRTQps_*.csv')))
    parts = []
    for f in files:
        for ch in pd.read_csv(f, usecols=['timestamp', 'msinstanceid',
                                          'metric', 'value'],
                              chunksize=CHUNK,
                              dtype={'msinstanceid': 'str', 'metric': 'category',
                                     'value': 'float32', 'timestamp': 'int64'}):
            ch = ch[ch['metric'].isin(MCR_METRICS)]
            if len(ch):
                parts.append(ch.groupby(['timestamp', 'msinstanceid'],
                                        observed=True)['value'].sum().reset_index())
    df = pd.concat(parts, ignore_index=True)
    return (df.groupby(['timestamp', 'msinstanceid'], observed=True)['value']
            .sum().reset_index().rename(columns={'value': 'workload'}))


def partial_corr(x, y, Z):
    """Tuong quan giua x va y sau khi khu tuyen tinh anh huong cua Z."""
    if len(x) < 30:
        return np.nan
    Z = np.asarray(Z, float)
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def run():
    print("=== topology (de loai cap co canh call-graph) ===")
    edges = build_topology()
    edge_set = set()
    if edges is not None and not edges.empty:
        edge_set = {(r.parent, r.child) for r in edges.itertuples()}
        edge_set |= {(c, p) for p, c in edge_set}
    print(f"  {len(edge_set)//2} canh (ca hai chieu)")

    print("=== loading instance-level ===")
    res = load_instance_resource()
    wl = load_instance_workload()
    print(f"  resource {len(res):,} dong | workload {len(wl):,} dong")

    df = res.merge(wl, on=['timestamp', 'msinstanceid'], how='inner')
    print(f"  ghep: {len(df):,} dong, {df.msinstanceid.nunique():,} instance")
    if df.empty:
        return pd.DataFrame()

    # Giu instance co du diem. LAY MAU THEO NODE, khong ngau nhien toan cuc:
    # voi ~1300 node, hai instance boc ngau nhien gan nhu khong bao gio trung
    # node (lan chay truoc chi duoc 29 cap dong-node tren 4000 cap khac-node).
    # Uu tien cac node host NHIEU instance de sinh du cap trong-node.
    cnt = df.groupby('msinstanceid').size()
    keep = set(cnt[cnt >= MIN_POINTS].index)
    print(f"  instance >= {MIN_POINTS} diem: {len(keep):,}")
    df = df[df.msinstanceid.isin(keep)]

    inst_meta = (df.groupby('msinstanceid')
                 .agg(nodeid=('nodeid', 'first'), msname=('msname', 'first')))
    by_node = inst_meta.groupby('nodeid').apply(
        lambda g: list(g.index)).to_dict()
    multi = {n: v for n, v in by_node.items() if len(v) >= 2}
    print(f"  node host >= 2 instance: {len(multi):,} / {len(by_node):,}")

    # chon instance tu cac node dong duc truoc, roi bu them de co nhom khac-node
    chosen = []
    for n, v in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        chosen.extend(v)
        if len(chosen) >= MAX_INSTANCES:
            break
    if len(chosen) < MAX_INSTANCES:
        rest = [i for i in inst_meta.index if i not in set(chosen)]
        chosen += list(RNG.choice(rest, min(len(rest), MAX_INSTANCES - len(chosen)),
                                  replace=False))
    chosen = set(chosen)
    df = df[df.msinstanceid.isin(chosen)]

    meta = inst_meta.loc[sorted(chosen)].to_dict('index')
    series = {k: g.set_index('timestamp')[['instance_cpu_usage', 'workload']]
              for k, g in df.groupby('msinstanceid')}
    insts = sorted(series)
    print(f"  dung {len(insts):,} instance tren "
          f"{len({meta[i]['nodeid'] for i in insts}):,} node")

    def _ok(a, b):
        sa, sb = meta[a]['msname'], meta[b]['msname']
        return sa != sb and (sa, sb) not in edge_set

    # nhom DONG-NODE: duyet cap trong tung node
    same = []
    for n, v in multi.items():
        v = [i for i in v if i in series]
        for a, b in itertools.combinations(v, 2):
            if _ok(a, b):
                same.append((a, b))
        if len(same) >= MAX_PAIRS:
            break
    same = same[:MAX_PAIRS]

    # nhom KHAC-NODE: boc ngau nhien, ep khac nodeid
    random.seed(SEED)
    diff, tries = [], 0
    while len(diff) < MAX_PAIRS and tries < 400_000:
        tries += 1
        a, b = random.sample(insts, 2)
        if meta[a]['nodeid'] != meta[b]['nodeid'] and _ok(a, b):
            diff.append((a, b))
    print(f"  cap dong-node {len(same):,} | khac-node {len(diff):,}")

    rows = []
    for label, pairs in (('same_node', same), ('diff_node', diff)):
        for a, b in pairs:
            ga, gb = series[a], series[b]
            idx = ga.index.intersection(gb.index)
            if len(idx) < MIN_POINTS:
                continue
            ca = ga.loc[idx, 'instance_cpu_usage'].values
            cb = gb.loc[idx, 'instance_cpu_usage'].values
            Z = np.column_stack([ga.loc[idx, 'workload'].values,
                                 gb.loc[idx, 'workload'].values])
            pc = partial_corr(ca, cb, Z)
            if np.isfinite(pc):
                rows.append(dict(group=label, n=len(idx), partial_corr=pc,
                                 abs_pc=abs(pc)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'alibaba_confounder_test.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    print(f"\n{'='*72}")
    print("  |Tuong quan rieng phan| CPU_i ~ CPU_j  SAU khi khu workload ca hai")
    print("  (chi cap khac service, khong co canh call-graph)")
    print(f"{'='*72}")
    g = d.groupby('group').agg(n=('abs_pc', 'size'), median=('abs_pc', 'median'),
                               mean=('abs_pc', 'mean'),
                               p75=('abs_pc', lambda x: x.quantile(0.75))).round(4)
    print(g.to_string())

    a = d[d.group == 'same_node'].abs_pc.dropna()
    b = d[d.group == 'diff_node'].abs_pc.dropna()
    if len(a) < 30 or len(b) < 30:
        print("\n  khong du mau de kiem dinh")
        return
    u, p = stats.mannwhitneyu(a, b, alternative='greater')
    auc = u / (len(a) * len(b))
    print(f"\n  Mann-Whitney (dong-node > khac-node, mot phia):")
    print(f"    p = {p:.4g} | common-language effect = {auc:.4f}")
    print(f"    median: dong-node {a.median():.4f} vs khac-node {b.median():.4f}")

    print(f"\n{'='*72}")
    if p < 0.05 and auc > 0.56:
        print("  CO bang chung confounder ha tang tren Alibaba.")
        print("  => do() CO viec de lam o day. Ket luan RQ5(b) phai doi thanh:")
        print("     'do() rong tren topology nho, co noi dung tren ha tang dung chung'.")
        print("  => Buoc tiep: chay hieu chinh backdoor xem co cai thien that khong.")
    else:
        print("  KHONG du bang chung confounder ha tang, ngay ca o quy mo 90k container.")
        print("  => Ket luan RQ5(b) duoc cung co bang hai he khac nhau hai bac quy mo.")
    print(f"{'='*72}")


if __name__ == '__main__':
    report(run())
