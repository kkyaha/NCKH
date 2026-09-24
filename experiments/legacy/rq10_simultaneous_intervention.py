# -*- coding: utf-8 -*-
"""
Huong 2 -- Can thiep DONG THOI len to tien va hau due: cho do() co noi dung
============================================================================
Ba thi nghiem truoc (RQ7/RQ8/RQ9) cho thay o cau hinh hien tai -- MOT can
thiep tai node goc, khong confounder do duoc -- toan tu do() quy ve dieu kien
hoa va khong mang lai loi the uoc luong nao. Ket luan do dung, nhung no gan
voi CAU HINH, khong phai voi toan tu.

Day la truong hop do() CO noi dung, va no khac han ba thi nghiem tren o mot
diem quan trong: ket qua la HE QUA DAI SO, chung minh duoc tren giay, khong
phai mot gia thuyet thuc nghiem co the that bai.

Van de
------
Requirement thuc te thuong ngu y NHIEU thay doi cung luc, trong do co ca mot
service thuong nguon VA mot service ha nguon cua no. Vi du "them khuyen mai
Black Friday" vua tang tai o front-end, vua thay doi truc tiep khoi luong xu
ly o orders.

Truncated Factorization cua Pearl:

    P(v | do(W_A = a, W_D = d)) = prod_{j not in {A,D}} P(x_j | pa_j)

Diem then chot: co che f_D sinh ra W_D bi LOAI BO HOAN TOAN. Sau can thiep,
W_D = d CHINH XAC, bat ke gia tri cua A. Can thiep o to tien KHONG lan truyen
qua D nua -- no bi cat dut tai D.

Mot pipeline noi chuoi hoi quy (hoac chinh pipeline nay neu mo rong mot cach
tu nhien, vi no tham so hoa theo delta_pct tuong doi so voi baseline) se lam
khac: lan truyen delta cua A qua f_D roi ap delta cua D LEN TREN gia tri da
lan truyen. Ket qua la DEM HAI LAN.

Menh de (truong hop tuyen tinh, chung minh duoc)
------------------------------------------------
Gia su co che Tier-1 la W_D = beta * W_A + beta0, baseline (a0, d0) voi
d0 = beta*a0 + beta0. Can thiep dong thoi delta_A, delta_D:

    dung   : W_D = d0 * (1 + delta_D)
    naive  : W_D = (beta*a0*(1 + delta_A) + beta0) * (1 + delta_D)
                 = (d0 + beta*a0*delta_A) * (1 + delta_D)

    sai so tuyet doi = beta * a0 * delta_A * (1 + delta_D)
    sai so tuong doi = (beta * a0 / d0) * delta_A

Tuc sai so tuong doi = (ti le workload cua D duoc giai thich boi A) x delta_A.
Khi A giai thich phan lon tai cua D (beta*a0 ~ d0), sai so tuong doi ~ delta_A:
mot can thiep +50% o to tien lam uoc luong tai D vuot 50%.

Script nay (1) kiem tra cong thuc tren bang so hoc trong DAG DA FIT, va (2) do
sai so lan truyen xuong cac metric tai nguyen ha nguon.

Output: data/processed/scm_results/rq10_simultaneous_intervention.csv
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))

from capacity_agent import CapacityAgent  # noqa: E402

N_SAMPLES = 400
DELTAS_A = [10.0, 25.0, 50.0]
DELTAS_D = [10.0, 25.0]


def _tier1_pairs(model, df):
    """Cap (A, D) sao cho A -> D la canh Tier-1 (workload -> workload)."""
    g = model.graph
    out = []
    for u, v in g.edges():
        if u.endswith('_workload') and v.endswith('_workload') \
                and u in df.columns and v in df.columns:
            out.append((u, v))
    return out


def _mean_of(samples, col):
    return float(samples[col].mean()) if col in samples.columns else np.nan


def run():
    cap = CapacityAgent(llm=None, auto_train=True)
    model, df = cap.global_dag_model, cap.df_baseline
    if model is None or df is None:
        print("khong train duoc Global DAG")
        return pd.DataFrame()

    pairs = _tier1_pairs(model, df)
    print(f"Global DAG: {model.graph.number_of_nodes()} nodes, "
          f"{model.graph.number_of_edges()} edges | {len(pairs)} canh Tier-1\n")

    # cac cot resource ha nguon de do anh huong
    res_cols = [c for c in df.columns
                if c.endswith(('_cpu', '_mem')) and c in model.graph.nodes()]

    rows = []
    for wA, wD in pairs:
        a0, d0 = float(df[wA].mean()), float(df[wD].mean())
        if a0 <= 0 or d0 <= 0:
            continue
        # he so Tier-1 da fit: beta = dW_D/dW_A (uoc luong bang sai phan huu han
        # tren chinh co che da fit, khong gia dinh dang ham)
        try:
            mech = model.causal_mechanism(wD)
            pm = mech.prediction_model
            eps = max(1e-6, 0.01 * a0)
            # co che cua wD co the co nhieu cha; danh gia tai baseline cac cha
            parents = sorted(model.graph.predecessors(wD))
            base = np.array([[float(df[p].mean()) for p in parents]])
            idx = parents.index(wA)
            bump = base.copy(); bump[0, idx] += eps
            beta = float((pm.predict(bump).ravel()[0] - pm.predict(base).ravel()[0]) / eps)
        except Exception:
            continue

        # Neo cua cong thuc phai la f_D(baseline), khong phai trung binh thuc
        # nghiem cua W_D: hai gia tri nay lech nhau mot chut (intercept + trung
        # binh nhieu), va dung sai neo se lam menh de lech mot luong khong lien
        # quan gi den hieu ung dang do.
        d0_fit = float(pm.predict(base).ravel()[0])
        for dA in DELTAS_A:
            for dD in DELTAS_D:
                tgt_A = a0 * (1 + dA / 100.0)
                tgt_D = d0_fit * (1 + dD / 100.0)

                # (1) DUNG: do() dong thoi -- co che cua D bi loai bo
                s_correct = gcm.interventional_samples(
                    model,
                    interventions={wA: lambda x, w=tgt_A: w,
                                   wD: lambda x, w=tgt_D: w},
                    num_samples_to_draw=N_SAMPLES)

                # (2) NAIVE: lan truyen A truoc, doc W_D thu duoc, roi ap delta
                #     cua D LEN TREN -- dung cach mot pipeline noi chuoi lam.
                #     Tinh TIEN DINH qua chinh co che da fit, KHONG lay trung binh
                #     mau: trung binh Monte Carlo tren N mau them nhieu lay mau lam
                #     menh de dai so khong kiem tra duoc chinh xac. E[W_D | do(W_A)]
                #     = f_D(pa) voi ANM, nen day la gia tri dung.
                bump_A = base.copy()
                bump_A[0, idx] = tgt_A
                wD_propagated = float(pm.predict(bump_A).ravel()[0])
                tgt_D_naive = wD_propagated * (1 + dD / 100.0)
                s_naive = gcm.interventional_samples(
                    model,
                    interventions={wA: lambda x, w=tgt_A: w,
                                   wD: lambda x, w=tgt_D_naive: w},
                    num_samples_to_draw=N_SAMPLES)

                # sai so du doan boi menh de
                pred_abs_err = beta * a0 * (dA / 100.0) * (1 + dD / 100.0)
                pred_rel_err = (beta * a0 / d0_fit) * (dA / 100.0) * 100.0

                row = {
                    'ancestor': wA, 'descendant': wD,
                    'beta_fitted': round(beta, 4),
                    'baseline_A': round(a0, 3), 'baseline_D': round(d0_fit, 4),
                    'delta_A_pct': dA, 'delta_D_pct': dD,
                    'wD_correct': round(tgt_D, 4),
                    'wD_naive': round(tgt_D_naive, 4),
                    'wD_abs_err': round(tgt_D_naive - tgt_D, 4),
                    'wD_rel_err_pct': round(100 * (tgt_D_naive - tgt_D) / tgt_D, 2),
                    'predicted_abs_err': round(pred_abs_err, 4),
                    'predicted_rel_err_pct': round(pred_rel_err, 2),
                }
                # anh huong xuong resource ha nguon
                errs = []
                for rc in res_cols:
                    c, n = _mean_of(s_correct, rc), _mean_of(s_naive, rc)
                    if np.isfinite(c) and np.isfinite(n) and abs(c) > 1e-9:
                        errs.append(abs(n - c) / abs(c) * 100)
                row['downstream_mean_rel_err_pct'] = round(float(np.mean(errs)), 3) if errs else np.nan
                row['downstream_max_rel_err_pct'] = round(float(np.max(errs)), 3) if errs else np.nan
                rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'rq10_simultaneous_intervention.csv'), index=False)
    return out


def report(d):
    if d.empty:
        print("khong co du lieu")
        return
    print("=" * 80)
    print("  Kiem tra menh de: sai so tuong doi tai D = (beta*a0/d0) * delta_A")
    print("=" * 80)
    d['formula_gap'] = (d.wD_rel_err_pct - d.predicted_rel_err_pct).abs()
    print(f"  n = {len(d)} | do lech giua do duoc va cong thuc: "
          f"max = {d.formula_gap.max():.4f} (percentage points)")
    print(f"  -> cong thuc {'KHOP' if d.formula_gap.max() < 0.5 else 'KHONG khop'} "
          f"voi DAG da fit\n")

    print("=" * 80)
    print("  Sai so cua phep noi chuoi hoi quy, theo delta cua to tien")
    print("=" * 80)
    g = d.groupby('delta_A_pct').agg(
        n=('wD_rel_err_pct', 'size'),
        wD_rel_err=('wD_rel_err_pct', 'median'),
        downstream_mean=('downstream_mean_rel_err_pct', 'median'),
        downstream_max=('downstream_max_rel_err_pct', 'max')).round(2)
    print(g.to_string())

    print("\n" + "=" * 80)
    print("  Cap Tier-1 bi anh huong nang nhat (delta_A = 50%)")
    print("=" * 80)
    w = d[d.delta_A_pct == 50].nlargest(6, 'wD_rel_err_pct')[
        ['ancestor', 'descendant', 'beta_fitted', 'wD_correct', 'wD_naive',
         'wD_rel_err_pct', 'downstream_max_rel_err_pct']]
    print(w.to_string(index=False))

    print("\n  Doc ket qua: sai so nay KHONG the giam bang cach fit tot hon. No la")
    print("  he qua cua viec khong loai bo co che tai node bi can thiep -- dung")
    print("  dieu Truncated Factorization quy dinh. Mot pipeline khong co toan tu")
    print("  do() se mac loi nay moi khi requirement ngu y can thiep dong thoi")
    print("  len mot to tien VA mot hau due cua no.")


if __name__ == '__main__':
    report(run())
