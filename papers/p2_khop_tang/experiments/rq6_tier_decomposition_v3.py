# -*- coding: utf-8 -*-
"""RQ6 v3 — PHAN RA TANG tren BA he, MOI node `_cpu`, va KIEM NGUONG NGOAI MAU.

Vi sao phai co v3 (ba ly do, ly do dau la mot loi phai sua):

  (1) v2 chay ngay 12/09, TRUOC khi `src/graph/trainticket_scm_edges.json` duoc
      sinh lai ngay 18/09. Tap canh tranh chap da DOI HOAN TOAN: 4 node dich cua
      v2 (ts-order / ts-seat / ts-travel / ts-user) gio KHONG con cha tranh chap
      nao, va 8 node khac thi co. Giao nhau cua hai tap node = RONG. Nen moi con
      so cua v2 -- ke ca Pearson 0,957 va nguong 50% -- khong tai lap duoc tren
      trang thai hien tai cua repo. Phai do lai.

  (2) v2 chi co 10 node tren 2 he. v3 chay MOI node `_cpu` co cha, tren 3 he.

  (3) Nguong 50% cua v2 duoc fit TRONG MAU roi bao nhu mot nguong. v3 kiem
      NGOAI MAU bang leave-one-system-out: fit nguong tren hai he, cham tren he
      thu ba. Chi khi do no moi la mot nguong van hanh.

Thiet ke giu NGUYEN cua v2 de so sanh duoc:
  tier1_driven : do(gateway_workload = dose x trung binh)  -> dap an = 'tier1'
  tier2_driven : gateway giu nguyen, gia tri CUA CHINH node bi nhieu +k sigma
                 -> dap an = 'tier2'
Dap an dung do XAY DUNG, khong phai do suy luan.

Khac v2 mot diem co y: v3 dung CapacityAgent cho CA BA he (v2 dung
evaluation_suite cho Sock Shop va CapacityAgent cho Train Ticket). Thong nhat
duong ma nghia la so sanh giua ba he hop le; doi lai, so cua Sock Shop khong
so truc tiep voi v2 duoc.

Chay:  python papers/p2_khop_tang/experiments/rq6_tier_decomposition_v3.py --repeats=5
Ra:    data/processed/scm_results/rq6_tier_decomposition_v3.csv
       data/processed/scm_results/rq6_tier_decomposition_v3_nodes.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TQDM_DISABLE'] = '1'

_P = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/ -- KHONG phu thuoc do sau
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
assert os.path.isdir(os.path.join(_P, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
for p in (os.path.join(_P, 'src'), os.path.join(_P, 'src', 'agents'), os.path.join(_P, 'src', 'scm')):
    sys.path.insert(0, p)
RES = os.path.join(_P, 'data', 'processed', 'scm_results')

from dowhy import gcm  # noqa: E402
from capacity_agent import CapacityAgent  # noqa: E402

RESOURCE_SUFFIXES = ('_cpu', '_mem', '_latency-50')
HE = [('SockShop', 'sockshop', 'front-end_workload'),
      ('OnlineBoutique', 'onlineboutique', 'frontend_workload'),
      ('TrainTicket', 'trainticket', 'ts-preserve-service_workload')]


def phan_loai(contribs: dict, target: str):
    """Chia dong gop theo tung to tien thanh (tier1, tranh chap, tier2) -- nguyen van v2."""
    t1 = bp = t2 = 0.0
    for node, arr in contribs.items():
        val = float(arr[0]) if len(arr) > 0 else 0.0
        if node == target:
            t2 += val
        elif node.endswith('_workload'):
            t1 += val
        elif node.endswith(RESOURCE_SUFFIXES):
            bp += val
    return t1, bp, t2


def chay_he(ten, system_type, gateway, n_repeats, dose=2.5, perturb_std=5.0):
    print(f'\n{"=" * 80}\n  {ten} — huan luyen DAG\n{"=" * 80}', flush=True)
    cap = CapacityAgent(system_type=system_type, auto_train=False)
    cap.train_accurate_path()
    model, df, g = cap.global_dag_model, cap.global_df_baseline, cap.dag_graph
    if gateway not in df.columns:
        print(f'  [BO QUA] khong co node gateway {gateway}')
        return pd.DataFrame()

    targets = sorted(n for n in g.nodes if n.endswith('_cpu') and g.in_degree(n) > 0
                     and n in df.columns)
    print(f'  {len(targets)} node dich | gateway = {gateway}', flush=True)

    base = df[gateway].mean()
    std_map, mean_map = df.std(), df.mean()
    rows = []
    for j, target in enumerate(targets, 1):
        for i in range(n_repeats):
            # --- kich ban A: tier1_driven (can thiep nhu cau that o gateway) ---
            try:
                smp = gcm.interventional_samples(
                    model, interventions={gateway: lambda x, w=base * dose: w},
                    num_samples_to_draw=200)
                row = mean_map.copy()
                for c in smp.columns:
                    row[c] = smp[c].mean()
                ca = gcm.attribute_anomalies(model, target_node=target,
                                             anomaly_samples=pd.DataFrame([row]),
                                             attribute_mean_deviation=True,
                                             num_distribution_samples=500)
                t1, bp, t2 = phan_loai(ca, target)
                strict = 'tier1' if abs(t1) > abs(t2) else 'tier2'
                upstream = 'upstream' if abs(t1 + bp) > abs(t2) else 'local'
            except Exception as e:
                t1 = bp = t2 = np.nan
                strict = upstream = f'ERROR:{str(e)[:40]}'
            rows.append(dict(he=ten, target=target, repeat=i, scenario='tier1_driven',
                             tier1=t1, backpressure=bp, tier2=t2,
                             strict_correct=strict == 'tier1', upstream_correct=upstream == 'upstream'))

            # --- kich ban B: tier2_driven (gateway giu nguyen, nhieu CHINH node) ---
            try:
                row = mean_map.copy()
                row[gateway] = base
                row[target] = mean_map[target] + perturb_std * std_map[target]
                cb = gcm.attribute_anomalies(model, target_node=target,
                                             anomaly_samples=pd.DataFrame([row]),
                                             attribute_mean_deviation=True,
                                             num_distribution_samples=500)
                t1, bp, t2 = phan_loai(cb, target)
                strict = 'tier1' if abs(t1) > abs(t2) else 'tier2'
                upstream = 'upstream' if abs(t1 + bp) > abs(t2) else 'local'
            except Exception as e:
                t1 = bp = t2 = np.nan
                strict = upstream = f'ERROR:{str(e)[:40]}'
            rows.append(dict(he=ten, target=target, repeat=i, scenario='tier2_driven',
                             tier1=t1, backpressure=bp, tier2=t2,
                             strict_correct=strict == 'tier2', upstream_correct=upstream == 'local'))
        print(f'  [{ten}] {j}/{len(targets)} {target}', flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=5)
    a = ap.parse_args()

    D = pd.concat([chay_he(t, s, gw, a.repeats) for t, s, gw in HE], ignore_index=True)
    D.to_csv(os.path.join(RES, 'rq6_tier_decomposition_v3.csv'), index=False)

    # do chinh xac theo node, tren kich ban tier1_driven (kich ban kho)
    t1 = D[D.scenario == 'tier1_driven']
    N = (t1.groupby(['he', 'target'])
         .agg(n=('strict_correct', 'size'),
              strict_acc=('strict_correct', lambda s: 100 * s.mean()),
              upstream_acc=('upstream_correct', lambda s: 100 * s.mean())).reset_index())

    # ghep voi TI PHAN NHU CAU do tu HE SO NNLS (doc lap Shapley)
    mp = os.path.join(RES, 'rq6_mechanism_three_systems_nodes.csv')
    if os.path.exists(mp):
        M = pd.read_csv(mp)[['he', 'dich', 'ti_phan_nhu_cau', 'ti_phan_tranh_chap',
                             'ti_phan_phan_du', 'r2', 'co_tranh_chap']]
        N = N.merge(M, left_on=['he', 'target'], right_on=['he', 'dich'], how='left').drop(columns='dich')
    N.to_csv(os.path.join(RES, 'rq6_tier_decomposition_v3_nodes.csv'), index=False)

    with pd.option_context('display.width', 220, 'display.max_rows', 200,
                           'display.float_format', lambda v: f'{v:,.2f}'):
        print(f'\n{"=" * 100}\n  DO CHINH XAC QUY GAN theo node (kich ban tier1_driven)\n{"=" * 100}')
        print(N.sort_values(['he', 'strict_acc']).to_string(index=False))
        print(f'\n{"=" * 100}\n  TONG HOP THEO HE\n{"=" * 100}')
        print(N.groupby('he').agg(n_node=('target', 'size'),
                                  strict_trung_vi=('strict_acc', 'median'),
                                  upstream_trung_vi=('upstream_acc', 'median'),
                                  nhu_cau_trung_vi=('ti_phan_nhu_cau', 'median')).to_string())

    # ---- TIEU CHI KHA NHAN: ti phan nhu cau co du doan duoc do chinh xac? ----
    ok = N.dropna(subset=['ti_phan_nhu_cau', 'strict_acc'])
    if len(ok) > 3:
        r, p = pearsonr(ok.ti_phan_nhu_cau, ok.strict_acc)
        print(f'\n[TRONG MAU] Pearson(ti phan nhu cau, strict_acc) = {r:+.3f}  (p = {p:.3g}, n = {len(ok)})')
        for ten_dd, cot in [('ti phan nhu cau', 'ti_phan_nhu_cau'), ('R^2 co che', 'r2'),
                            ('ti phan tranh chap', 'ti_phan_tranh_chap')]:
            rr, pp = pearsonr(ok[cot], ok.strict_acc)
            print(f'   {ten_dd:22s} r = {rr:+.3f}  (p = {pp:.3g})')

        # ---- KIEM NGOAI MAU: leave-one-system-out ----
        print(f'\n{"=" * 100}\n  KIEM NGOAI MAU — leave-one-system-out cho NGUONG\n{"=" * 100}')
        print('  fit nguong tren HAI he (quet 0..100, chon nguong toi da hoa do tach),')
        print('  roi cham tren he THU BA. Day moi la nguong van hanh.\n')
        for he_test in sorted(ok.he.unique()):
            tr, te = ok[ok.he != he_test], ok[ok.he == he_test]
            if len(tr) < 3 or len(te) < 2:
                continue
            best, best_nguong = -np.inf, np.nan
            for ng in np.arange(5, 100, 2.5):
                hi, lo = tr[tr.ti_phan_nhu_cau > ng], tr[tr.ti_phan_nhu_cau <= ng]
                if len(hi) == 0 or len(lo) == 0:
                    continue
                sep = hi.strict_acc.mean() - lo.strict_acc.mean()
                if sep > best:
                    best, best_nguong = sep, ng
            if np.isnan(best_nguong):
                print(f'  {he_test:15s} khong fit duoc nguong tren hai he con lai')
                continue
            hi, lo = te[te.ti_phan_nhu_cau > best_nguong], te[te.ti_phan_nhu_cau <= best_nguong]
            print(f'  giu lai {he_test:15s} | nguong fit = {best_nguong:5.1f}%  '
                  f'(do tach tren tap fit = {best:+5.1f} diem)')
            print(f'      tren nguong: n = {len(hi):2d}, do chinh xac TB = '
                  f'{hi.strict_acc.mean() if len(hi) else float("nan"):6.1f}%')
            print(f'      duoi nguong: n = {len(lo):2d}, do chinh xac TB = '
                  f'{lo.strict_acc.mean() if len(lo) else float("nan"):6.1f}%')
            if len(hi) and len(lo):
                print(f'      -> do tach NGOAI MAU = {hi.strict_acc.mean() - lo.strict_acc.mean():+.1f} diem')

    print(f'\n-> {RES}/rq6_tier_decomposition_v3.csv  ({len(D)} dong)')
    print(f'-> {RES}/rq6_tier_decomposition_v3_nodes.csv  ({len(N)} dong)')


if __name__ == '__main__':
    main()
