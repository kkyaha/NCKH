# -*- coding: utf-8 -*-
"""RQ6 — KHU VONG TRON cho ket qua dinh vi tang workload, va chay tren CA HAI he.

VAN DE VONG TRON: ket qua "xep hang theo sigma-shift tang workload dat AUC 0.991"
duoc cham diem voi tap `reachable` DINH NGHIA BANG DO THI GOI -- ma canh
workload->workload cua DAG lai DUOC DUNG TU chinh do thi goi do. Nen mot phan ket
qua la he qua cua dinh nghia.

BA PHEP CHAM, tu vong tron nhat den sach nhat:
  (A) reachable theo DO THI GOI            -- vong tron cau truc
  (B) reachable theo DU LIEU QUAN SAT      -- service co workload tuong quan voi
      workload gateway trong du lieu nen (khong dung do thi)
  (C) Spearman(dich chuyen can thiep, tuong quan quan sat)  -- du lieu doi du lieu,
      khong nguong, khong do thi o ca hai phia. Day la phep sach nhat.

Doi chieu trong moi phep: tang WORKLOAD vs tang CPU. Cung DAG, cung can thiep.

Chay:  python experiments/legacy/rq6_localization_decircularized.py [--repeats=10]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
sys.path[:0] = [os.path.join(_P, 'src'), os.path.join(_P, 'src', 'agents'), os.path.join(_P, 'src', 'scm')]
from dowhy import gcm                                              # noqa: E402
from capacity_agent import CapacityAgent                           # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402

RES = os.path.join(_P, 'data', 'processed', 'scm_results')
SYS = {
    'SockShop':    dict(system_type='sockshop',    gw='front-end',           graph='sockshop_agent_graph.json'),
    'OnlineBoutique': dict(system_type='onlineboutique', gw='frontend', graph='onlineboutique_agent_graph.json'),
    'TrainTicket': dict(system_type='trainticket', gw='ts-preserve-service', graph='trainticket_agent_graph.json'),
}
DOSE = 2.5
CORR_MIN = 0.30        # nguong cho phep cham (B); phep (C) khong dung nguong


def run_system(name, cfg, n_repeats):
    print('\n' + '=' * 78)
    print(f'  {name}  |  gateway = {cfg["gw"]}  |  do(workload = {DOSE}x)')
    print('=' * 78)
    cap = CapacityAgent(system_type=cfg['system_type'], auto_train=False)
    cap.train_accurate_path()
    model, df = cap.global_dag_model, cap.global_df_baseline
    INJ = f'{cfg["gw"]}_workload'
    if INJ not in df.columns:
        print(f'  [BO QUA] khong co {INJ}'); return None

    G = json.load(open(os.path.join(_P, 'src', 'graph', cfg['graph']), encoding='utf-8'))
    svcs = {n['id'] if isinstance(n, dict) else n for n in G['nodes']}
    gs = nx.DiGraph(); gs.add_nodes_from(svcs)
    gs.add_edges_from([(e.get('source', e.get('from')), e.get('target', e.get('to'))) for e in G['edges']])
    reach_graph = nx.descendants(gs, cfg['gw']) if cfg['gw'] in gs else set()

    cand = sorted({c.rsplit('_', 1)[0] for c in df.columns if c.endswith('_workload')}
                  & {c.rsplit('_', 1)[0] for c in df.columns if c.endswith('_cpu')} - {cfg['gw']})

    # (B)/(C): tuong quan QUAN SAT giua workload cua service va workload gateway
    obs = {}
    for s in cand:
        col = f'{s}_workload'
        x, y = df[INJ].to_numpy(float), df[col].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        obs[s] = abs(float(np.corrcoef(x[m], y[m])[0, 1])) if m.sum() > 10 and y[m].std() > 0 else 0.0
    reach_obs = {s for s, v in obs.items() if v >= CORR_MIN}
    print(f'[Setup] {len(cand)} ung vien | reachable theo DO THI: {len(reach_graph & set(cand))} '
          f'| theo QUAN SAT (|r|>={CORR_MIN}): {len(reach_obs)}')
    print(f'        hai tap trung nhau: {len((reach_graph & set(cand)) & reach_obs)} service')

    base = {c: (float(df[c].mean()), float(df[c].std())) for c in df.columns}
    tgt = base[INJ][0] * DOSE
    rows = []
    for i in range(n_repeats):
        s = gcm.interventional_samples(model, interventions={INJ: lambda x, w=tgt: w},
                                       num_samples_to_draw=200)
        for svc in cand:
            r = dict(system=name, repeat=i, service=svc,
                     reach_graph=svc in reach_graph, reach_obs=svc in reach_obs, obs_corr=obs[svc])
            for kind in ('workload', 'cpu'):
                col = f'{svc}_{kind}'
                if col in s.columns and col in base and base[col][1] > 0:
                    r[f'sig_{kind}'] = (float(s[col].mean()) - base[col][0]) / base[col][1]
                else:
                    r[f'sig_{kind}'] = np.nan
            rows.append(r)
    D = pd.DataFrame(rows)

    print(f'\n--- (A) cham theo DO THI GOI (vong tron cau truc) ---')
    _score(D, 'reach_graph')
    print(f'--- (B) cham theo DU LIEU QUAN SAT (|r| >= {CORR_MIN}) ---')
    _score(D, 'reach_obs')
    print('--- (C) Spearman(dich chuyen can thiep, tuong quan quan sat) — du lieu doi du lieu ---')
    for sig, lab in (('sig_workload', 'tang WORKLOAD'), ('sig_cpu', 'tang CPU')):
        per = [spearmanr(g[sig], g.obs_corr)[0] for _, g in D.groupby('repeat') if g[sig].notna().sum() > 5]
        print(f'    {lab:16s} rho = {np.nanmean(per):+.3f}  (trung binh {len(per)} repeat, '
              f'khoang {np.nanmin(per):+.2f}..{np.nanmax(per):+.2f})')
    return D


def _score(D, label_col):
    k = int(D.groupby('repeat')[label_col].sum().median())
    chance = 100.0 * D[label_col].mean()
    for sig, lab in (('sig_workload', 'tang WORKLOAD'), ('sig_cpu', 'tang CPU')):
        x = D.dropna(subset=[sig])
        if x[label_col].nunique() < 2 or k < 1:
            continue
        per = [100.0 * g.nlargest(k, sig)[label_col].mean() for _, g in x.groupby('repeat') if len(g) >= k]
        auc = roc_auc_score(x[label_col], x[sig])
        print(f'    {lab:16s} Precision@{k} = {np.mean(per):5.1f}%  (ngau nhien {chance:4.1f}%, '
              f'hon {np.mean(per)-chance:+5.1f} d)  AUC = {auc:.3f}')


if __name__ == '__main__':
    n = 10
    for a in sys.argv:
        if a.startswith('--repeats='):
            n = int(a.split('=', 1)[1])
    out = [d for nm, c in SYS.items() if (d := run_system(nm, c, n)) is not None]
    if out:
        p = os.path.join(RES, 'rq6_localization_decircularized.csv')
        pd.concat(out, ignore_index=True).to_csv(p, index=False)
        print(f'\n[OK] da ghi {p}')
