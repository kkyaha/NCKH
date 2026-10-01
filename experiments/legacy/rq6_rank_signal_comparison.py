# -*- coding: utf-8 -*-
"""RQ6 — DINH VI nguyen nhan lan truyen: xep hang theo TIN HIEU nao?

Cau hoi: Shapley credit la cong cu PHAN BO trach nhiem. Dinh vi (node nao nam ha
nguon cua nguyen nhan) la cau hoi KHAC. Bai nay do xem tin hieu nao dinh vi dung hon.

Bon tin hieu, cung mot lan can thiep do(ts-preserve-service_workload = 2.5x):
  1. sigma-shift cua node WORKLOAD  -- dich chuyen tren chinh kenh lan truyen nhu cau
  2. sigma-shift cua node CPU        -- dich chuyen o dai luong duoc quy gan
  3. pct-shift cua node CPU          -- tuong duong `change_pct` trong rq6_topology_check
  (doi chieu: shapley_contribution tu rq6_topology_check.csv)

Chi so: Precision@k voi k = so service reachable tu diem tiem tren DO THI GOI,
so voi muc ngau nhien = ti le reachable. MOT loi goi interventional_samples tra ve
MOI node, nen mo rong tu 4 target (rq6_propagation_path_diagnostic) len ca 27
service gan nhu khong ton them gi.

Chay:  python experiments/legacy/rq6_rank_signal_comparison.py [--repeats=10]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [os.path.join(_P, 'src'), os.path.join(_P, 'src', 'agents'), os.path.join(_P, 'src', 'scm')]
from dowhy import gcm                                              # noqa: E402
from capacity_agent import CapacityAgent                           # noqa: E402

RES = os.path.join(_P, 'data', 'processed', 'scm_results')
INJ_SVC = 'ts-preserve-service'
INJ = f'{INJ_SVC}_workload'
DOSE = 2.5


def main(n_repeats=10):
    print('=' * 78)
    print('  RQ6 — dinh vi nguyen nhan lan truyen: so sanh TIN HIEU XEP HANG')
    print(f'  do({INJ} = {DOSE}x) | {n_repeats} repeat | khong goi LLM')
    print('=' * 78)

    cap = CapacityAgent(system_type='trainticket', auto_train=False)
    cap.train_accurate_path()
    model, df = cap.global_dag_model, cap.global_df_baseline

    # reachable theo DO THI GOI (khong dung DAG da fit, de doc lap voi mo hinh)
    G = json.load(open(os.path.join(_P, 'src', 'graph', 'trainticket_agent_graph.json'), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    gsvc = nx.DiGraph(); gsvc.add_nodes_from(svcs)
    gsvc.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    reach = nx.descendants(gsvc, INJ_SVC)

    # service co CA node workload va node cpu trong DAG da fit
    cand = sorted({c.rsplit('_', 1)[0] for c in df.columns if c.endswith('_workload')}
                  & {c.rsplit('_', 1)[0] for c in df.columns if c.endswith('_cpu')} - {INJ_SVC})
    print(f'\n[Setup] {len(cand)} service ung vien | {sum(s in reach for s in cand)} reachable tren do thi goi')

    base_val = float(df[INJ].mean())
    tgt = base_val * DOSE
    print(f'[Setup] do({INJ}: {base_val:.4f} -> {tgt:.4f})')

    base = {c: (float(df[c].mean()), float(df[c].std())) for c in df.columns}
    rows = []
    for i in range(n_repeats):
        s = gcm.interventional_samples(model, interventions={INJ: lambda x, w=tgt: w},
                                       num_samples_to_draw=200)
        for svc in cand:
            r = dict(repeat=i, service=svc, reachable=svc in reach)
            for kind in ('workload', 'cpu'):
                col = f'{svc}_{kind}'
                if col not in s.columns or col not in base:
                    r[f'sig_{kind}'] = np.nan; r[f'pct_{kind}'] = np.nan; continue
                m, sd = base[col]
                iv = float(s[col].mean())
                r[f'sig_{kind}'] = (iv - m) / sd if sd > 0 else np.nan
                r[f'pct_{kind}'] = 100.0 * (iv - m) / m if m != 0 else np.nan
            rows.append(r)
        print(f'  [repeat {i+1}/{n_repeats}] xong')
    D = pd.DataFrame(rows)
    out = os.path.join(RES, 'rq6_rank_signal_comparison.csv')
    D.to_csv(out, index=False)
    print(f'\n[OK] da ghi {out}')

    k = int(D.groupby('repeat').reachable.sum().median())
    chance = 100.0 * D.reachable.mean()
    print(f'\n=== Precision@{k} | muc ngau nhien {chance:.1f}% ===')
    res = []
    for sig, lab in (('sig_workload', 'sigma-shift node WORKLOAD'),
                     ('sig_cpu', 'sigma-shift node CPU'),
                     ('pct_cpu', 'pct-shift node CPU (= change_pct)')):
        per = []
        for _, g in D.groupby('repeat'):
            gg = g.dropna(subset=[sig])
            if len(gg) < k: continue
            per.append(100.0 * gg.nlargest(k, sig).reachable.mean())
        if per:
            res.append(dict(tin_hieu=lab, precision=np.mean(per), hon_ngau_nhien=np.mean(per) - chance,
                            spread=f'{min(per):.0f}-{max(per):.0f}'))
    print(pd.DataFrame(res).round(1).to_string(index=False))

    from sklearn.metrics import roc_auc_score
    print('\n=== AUC (phan biet reachable vs unreachable) ===')
    for sig in ('sig_workload', 'sig_cpu', 'pct_cpu'):
        x = D.dropna(subset=[sig])
        if x.reachable.nunique() > 1:
            print(f'  {sig:14s} {roc_auc_score(x.reachable, x[sig]):.3f}')
    print('\n(doi chieu Shapley credit tu rq6_topology_check.csv: Precision@17 = 59.4-75.9%, AUC 0.478-0.678)')


if __name__ == '__main__':
    n = 10
    for a in sys.argv:
        if a.startswith('--repeats='):
            n = int(a.split('=', 1)[1])
    main(n)
