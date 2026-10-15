# -*- coding: utf-8 -*-
"""
BASELINE NGOAI SUY DUOC vs He A (A1): hoi quy tuyen tinh + DINH LUAT SU DUNG, muc PHAN QUYET
=============================================================================================
Vi sao file nay ton tai: `gradboost_vs_heA_verdict.py` do xong GradBoost va cho ket qua kha-thi-gia
100% o CA 3 bien the. Truy nguyen (do bang so, xem muc "KET QUA DA BIET" duoi) cho thay do KHONG
phai "He A thang gradient boosting" ma la mot tinh chat cau truc: CAY khong du bao ngoai dai dap
ung da thay, nen voi 6/7 node CPU, max(y) tren tap fit cua GB con THAP HON tran P99 ma luat quyet
dinh dung => KHONG MOT dau vao nao co the khien GB bao bao hoa CPU. Theo dung dinh nghia
"admissible evaluation" cua bai bao (paper_draft.tex, Definition 1), phep so do la VACUOUS doi voi
GB: khong kec cuc nao bac bo duoc "GB noi kha thi".

=> Baseline co kha nang THU THACH A1 phai la mo hinh NGOAI SUY DUOC. File nay do 4 cau hinh do:

  LR_biv        hoi quy tuyen tinh, X = workload cua CHINH service (= 'LinearReg' trong
                model_comparison.py: Pipeline(StandardScaler, LinearRegression))
  LR_multi      hoi quy tuyen tinh, X = vector workload CA 7 service -- CUNG thong tin lan truyen
                voi He A, nhung KHONG co rang buoc cha cua do thi. Day la phep thu CAU TRUC sach
                nhat trong ca repo: co che cua He A la LinearRegression(positive=True) tren DUNG
                tap cha cua DAG; LR_multi la cung ho ham, cung du lieu, nhung cho TAT CA 7 service
                lam bien doc lap. Chenh lech giua A_off va LR_multi = gia tri cua viec HAN CHE cha
                theo do thi goi that (khong phai gia tri cua "hoi quy" noi chung).
  UtilLaw       DINH LUAT SU DUNG (operational law, Lazowska et al. / Little): U_s = X_s * S_s, voi
                S_s = nhu cau phuc vu (CPU-giay / request) uoc luong bang hoi quy QUA GOC tren tap
                fit: S_s = sum(CPU*W)/sum(W^2). Day la BASELINE DA CONG BO duy nhat ap duoc o day
                ma KHONG can mo hinh hieu nang cua chinh tinh nang chua ton tai (khac Palladio/LQN
                -- xem paper muc Related Work). CHI noi ve CPU: metric mem/latency giu = baseline
                (khai bao ro, khong giau: nghia la UtilLaw khong the bao bao hoa mem/latency).
  UtilLaw_MM1   UtilLaw + tang do tre M/M/1: R_s = R0_s / (1 - W_s/C_s), C_s = P99(W_s)*1.5 -- DUNG
                quy uoc C cua chinh repo (src/scm/queueing_regressor.py: QueueingLatencyRegressor),
                de phan do tre khong bi loi the/bat loi do chon C khac. W_s >= C_s -> +inf.

TAT CA deu di qua DUNG MOT tang quyet dinh: agent.compute_headroom() + agent.summarize_headroom()
cua CapacityAgent. Thu duy nhat khac nhau giua cac cau hinh la BO DU BAO. Bao dam bang self-check
bat buoc (tai su dung gradboost_vs_heA_verdict.tu_kiem_tra_A1): tai lap 288/288 status cua A1 tu
chinh du bao cua A1 qua dict do script nay dong goi. Khong dat 288/288 -> raise, khong bao cao gi.

PHEP KIEM MOI (dong gop rieng cua file nay) -- "ADMISSIBILITY" TUNG BASELINE:
  Voi tung cau hinh, dem (a) so cap (diem, node) co du bao VUOT tran P99, va (b) so diem co status
  != SAFE. Neu (b) == 0 tren CA 288 diem => phep danh gia la VACUOUS doi voi cau hinh do (khong ket
  cuc nao bac bo duoc "noi kha thi"), va con so kha-thi-gia cua no KHONG duoc trich dan nhu mot so
  so sanh nang luc. Day chinh la cai bay GB da roi vao; kiem o day de khong lap lai.
  KEM THEO phep kiem TIEN NGHIEM (khong can du lieu test): max(y) tren tap fit vs tran P99 tung
  node -- voi mo hinh bi chan boi dai dap ung (cay), day la gioi han tren CUNG; voi mo hinh tuyen
  tinh thi khong (ngoai suy duoc), nen cot nay chi co y nghia doc hieu, khong phai tieu chi loai.

KET QUA DA BIET (de doi chieu, tu 2 script truoc -- khong tinh lai):
  A1    (correction BAT): kha-thi-gia 6,7% (doc long) / 3,4% (doc chat); bo-thieu 10,8% / 22,3%
  A_off (correction TAT): kha-thi-gia 79,2% / 16,1%;                      bo-thieu 0% / 0,7%
  GB_*                  : kha-thi-gia 100% / 99,3-100% -- VACUOUS, xem tren
  Tren 12 cell NGOAI tap fit cua correction: A1 dat 0/55 kha-thi-gia.

GIA DINH / GIOI HAN (khong giau):
  - Cung nguon du lieu (data/raw/SS-TRAIN), cung giao thuc OOD (sap theo workload cua target, 67%
    thap de fit) cho MOI cau hinh ML -- giong gradboost_vs_heA_verdict.py.
  - LR khong bi chan duong: du bao CPU co the am o tai rat thap. KHONG kep ve 0 (kep la mot lua
    chon thiet ke them); neu xay ra, so lan duoc DEM va bao cao.
  - UtilLaw uoc luong S_s tren CHINH tap fit (tai thap). Neu nhu cau phuc vu thuc te tang theo tai
    (tranh chap, GC), UtilLaw se du bao THAP hon thuc te -- day la han che da biet cua operational
    law, bao cao chu khong vá.
  - SS-TRAIN KHONG co nhan tinh nang/archetype, nen KHONG cau hinh nao o day hoc duoc hieu ung
    theo chain. Bac so CUNG THONG TIN voi chung la A_off, khong phai A1 (A1 biet chain tu bang
    archetype + hieu chinh rho tren SS-LIMITS). So sanh voi A1 van duoc bao cao, nhung phai doc kem
    chu y nay.

CHAY LAI:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
      papers/p1_du_phong/experiments/model_eval/linear_utillaw_vs_heA_verdict.py
RA:
  data/processed/scm_results/linear_utillaw_vs_heA_verdict_matrix.csv   (288 dong)
  data/processed/scm_results/linear_utillaw_vs_heA_summary.csv
  data/processed/scm_results/linear_utillaw_vs_heA_mcnemar.csv
  data/processed/scm_results/linear_utillaw_admissibility.csv
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = _HERE
while not os.path.isdir(os.path.join(BASE_DIR, 'src')) and BASE_DIR != os.path.dirname(BASE_DIR):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, _HERE)

from capacity_agent import CapacityAgent  # noqa: E402
from data_processor import load_multi_service_data  # noqa: E402
import evaluate_verdict_level_heA as ev_heA  # noqa: E402
import gradboost_vs_heA_verdict as gbm  # noqa: E402  (TAI SU DUNG: target 'stable', lan truyen, self-check, tong hop)

OUT_DIR = ev_heA.OUT_DIR
SS_TRAIN = ev_heA.SS_TRAIN
GRAPH_PATH = ev_heA.GRAPH_PATH
SERVICES = gbm.SERVICES

BASE_CONFIGS = ['LR_biv', 'LR_multi', 'UtilLaw', 'UtilLaw_MM1']
ALL_CONFIGS = ['A1', 'A_off'] + BASE_CONFIGS


# ======================================================================
# 1. FIT: LR (2 bien the) + nhu cau phuc vu S_s cho dinh luat su dung
# ======================================================================
def _lr():
    """Giong 'LinearReg' trong model_comparison.py (scaler + hoi quy tuyen tinh, khong rang buoc dau)."""
    return Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])


def fit_cac_bien_the(agent, df_multi, targets):
    """Fit LR_biv, LR_multi (1 model/target) + uoc luong S_s (nhu cau phuc vu) cho UtilLaw.

    S_s uoc luong bang hoi quy QUA GOC tren tap fit: S = sum(y*W)/sum(W^2) -- cung dang uoc luong
    voi rho_s trong capacity_agent.py (nhat quan quy uoc trong repo). Chi co nghia cho metric cpu.
    Tra ve (models_biv, models_multi, demand, fit_meta).
    """
    models_biv, models_multi, demand, fit_meta = {}, {}, {}, {}
    for (svc, metric_key, suf, col) in targets:
        df_fit, df_test, wl_cols = gbm.build_fit_test(df_multi, svc, col)
        X_multi = df_fit[wl_cols].values
        X_biv = df_fit[[f'{svc}_workload']].values
        y = df_fit[col].values

        m_biv = _lr().fit(X_biv, y)
        m_multi = _lr().fit(X_multi, y)
        models_biv[(svc, metric_key)] = m_biv
        models_multi[(svc, metric_key)] = m_multi

        w = df_fit[f'{svc}_workload'].values
        denom = float(np.sum(w * w))
        demand[(svc, metric_key)] = (float(np.sum(y * w) / denom) if denom > 0 else np.nan)

        fit_meta[(svc, metric_key)] = dict(
            wl_cols=wl_cols, n_fit=len(df_fit), n_test=len(df_test),
            biv_min=float(w.min()), biv_max=float(w.max()),
            y_min_fit=float(y.min()), y_max_fit=float(y.max()),
            # C cho M/M/1: DUNG quy uoc QueueingLatencyRegressor (P99 * 1.5) tren tap fit
            C_mm1=float(np.percentile(w, 99) * 1.5),
            R0=float(np.median(y)),   # do tre "khong tai" dai dien (chi dung cho metric latency)
        )
    return models_biv, models_multi, demand, fit_meta


# ======================================================================
# 2. DU BAO TAI 288 DIEM + DONG GOI cho compute_headroom/summarize_headroom
# ======================================================================
def du_bao_tai_cac_diem(agent, cell_df, A, targets, models_biv, models_multi, demand, fit_meta):
    """Moi dong cua A = 1 diem quyet dinh (cell, muc_tai). Dung CHINH vector workload lan truyen
    cua He A (gbm.lay_workload_lan_truyen -> agent._deterministic_forward) de khong cau hinh nao bi
    loi the ve nguon workload. Tra ve (DataFrame ket qua, DataFrame dem vuot-tran tung cau hinh)."""
    spec_by_cell = {r.cell: r for r in cell_df.itertuples()}
    # tran P99 tung (svc, metric): lay 1 lan tu headroom o can thiep 0% (khong doi theo diem)
    hr0 = agent.compute_headroom(agent.simulate_intervention('front-end', 0.0))
    tran = {(s, mk): hr0.get(s, {}).get(mk, {}).get('ceiling_p99') for s in SERVICES
            for mk in ('cpu', 'mem', 'latency')}

    rows, vuot_tran = [], {c: 0 for c in BASE_CONFIGS}
    am_count = {c: 0 for c in BASE_CONFIGS}
    for row in A.itertuples():
        spec = spec_by_cell[row.cell]
        target_wl = float(row.muc_tai) * (1.0 + spec.delta_scaled_pct / 100.0)
        wl_vec, _ = gbm.lay_workload_lan_truyen(agent, target_wl)
        x_multi = np.array([[wl_vec[s] for s in SERVICES]])

        sim = {c: {} for c in BASE_CONFIGS}
        for (svc, metric_key, suf, col) in targets:
            meta = fit_meta[(svc, metric_key)]
            base = float(agent.global_df_baseline[col].mean())
            own_wl = wl_vec[svc]

            preds = {
                'LR_biv': float(models_biv[(svc, metric_key)].predict([[own_wl]])[0]),
                'LR_multi': float(models_multi[(svc, metric_key)].predict(x_multi)[0]),
            }
            # --- dinh luat su dung: chi CPU; mem/latency giu baseline (khai bao o docstring) ---
            if metric_key == 'cpu':
                u = demand[(svc, metric_key)] * own_wl
                preds['UtilLaw'] = float(u)
                preds['UtilLaw_MM1'] = float(u)
            elif metric_key == 'latency':
                preds['UtilLaw'] = base
                C = meta['C_mm1']
                if C > 0 and own_wl < C:
                    preds['UtilLaw_MM1'] = float(meta['R0'] / (1.0 - own_wl / C))
                else:
                    preds['UtilLaw_MM1'] = float('inf')
            else:  # mem
                preds['UtilLaw'] = base
                preds['UtilLaw_MM1'] = base

            for cfg in BASE_CONFIGS:
                p = preds[cfg]
                sim[cfg].setdefault(svc, {})[f'{metric_key}_baseline'] = base
                sim[cfg].setdefault(svc, {})[f'{metric_key}_predicted'] = p
                t = tran[(svc, metric_key)]
                if t is not None and np.isfinite(p) and p > t:
                    vuot_tran[cfg] += 1
                elif not np.isfinite(p):
                    vuot_tran[cfg] += 1
                if p < 0:
                    am_count[cfg] += 1

        out = dict(cell=row.cell, muc_tai=row.muc_tai, target_wl=target_wl)
        for cfg in BASE_CONFIGS:
            headroom = agent.compute_headroom(sim[cfg])
            status, _ = agent.summarize_headroom(headroom)
            out[f'{cfg}_status'] = status
            out[f'{cfg}_kha_thi_long'] = ev_heA.feasible_long(status)
            out[f'{cfg}_kha_thi_chat'] = ev_heA.feasible_chat(status)
        rows.append(out)

    dem = pd.DataFrame([dict(cau_hinh=c, n_cap_vuot_tran=vuot_tran[c], n_du_bao_am=am_count[c])
                        for c in BASE_CONFIGS])
    return pd.DataFrame(rows), dem


# ======================================================================
# 3. PHEP KIEM ADMISSIBILITY (dong gop rieng cua file nay)
# ======================================================================
def kiem_admissibility(M, dem, targets, fit_meta, agent):
    """Mot cau hinh la ADMISSIBLE tren phep danh gia nay neu co IT NHAT MOT diem trong 288 ma no
    tra status != SAFE (tuc ton tai ket cuc bac bo duoc 'noi kha thi'). Neu khong -> VACUOUS, va
    con so kha-thi-gia cua no khong duoc doc nhu mot so so sanh nang luc.
    Kem cot tien nghiem: so node co max(y_fit) <= tran P99 (voi mo hinh bi chan boi dai dap ung
    thi day la chan tren cung; voi LR/UtilLaw thi khong, chi doc hieu)."""
    hr0 = agent.compute_headroom(agent.simulate_intervention('front-end', 0.0))
    n_chan = 0
    for (svc, mk, suf, col) in targets:
        t = hr0.get(svc, {}).get(mk, {}).get('ceiling_p99')
        if t is not None and fit_meta[(svc, mk)]['y_max_fit'] <= t:
            n_chan += 1

    rows = []
    for cfg in ALL_CONFIGS:
        st = M[f'{cfg}_status']
        n_khong_safe = int((st != 'SAFE').sum())
        d = dem[dem.cau_hinh == cfg]
        rows.append(dict(
            cau_hinh=cfg,
            n_diem_khong_SAFE=n_khong_safe,
            admissible=(n_khong_safe > 0),
            n_cap_du_bao_vuot_tran=(int(d.n_cap_vuot_tran.iloc[0]) if not d.empty else None),
            n_du_bao_am=(int(d.n_du_bao_am.iloc[0]) if not d.empty else None),
            n_node_co_max_y_fit_duoi_tran=n_chan,
            tong_node_stable=len(targets),
        ))
    return pd.DataFrame(rows)


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 112)
    print("  BASELINE NGOAI SUY DUOC (LinearReg, dinh luat su dung) vs He A (A1) -- muc PHAN QUYET")
    print("=" * 112)

    print("\nKhoi tao CapacityAgent (llm=None, SS-TRAIN) -- khong goi LLM/ParserAgent...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent = CapacityAgent(llm=None, data_dir=SS_TRAIN, graph_path=GRAPH_PATH, auto_train=True)
        bridge = ev_heA.HeABridge(agent)
    if not ev_heA._check_evidence_sanity(bridge):
        raise RuntimeError("Kiem tra doi chung evidence (ev_heA) KHONG khop -- dung.")
    print("  [OK] cau bac L->delta khop 10/10 phan quyet doi chung.")

    cell_df = ev_heA.nap_27_cell()
    with contextlib.redirect_stdout(buf):
        A = ev_heA.phan_A_quyet_dinh_tai_luoi_do(bridge, cell_df)
    assert len(A) == 288, f"ky vong 288 diem, duoc {len(A)}"
    print(f"  [OK] {len(A)} diem quyet dinh ({int((~A.that_kha_thi).sum())} chac chan KHONG kha thi / "
          f"{int(A.that_kha_thi.sum())} chac chan kha thi).")

    print("\n--- SELF-CHECK BAT BUOC (tai su dung gbm.tu_kiem_tra_A1): tai lap status A1 tu du bao A1 ---")
    with contextlib.redirect_stdout(buf):
        n_match, n_total, mismatches = gbm.tu_kiem_tra_A1(agent, cell_df, A)
    print(f"  {n_match}/{n_total} khop.")
    if n_match != n_total:
        for m in mismatches[:10]:
            print(f"    LECH cell={m[0]} muc_tai={m[1]} tu_tinh={m[2]} A1={m[3]}")
        raise RuntimeError("Self-check khong dat 288/288 -- DUNG, khong bao cao so sanh nao.")

    targets = gbm.lay_target_on_dinh(agent)
    print(f"\n--- {len(targets)} node 'stable' (chi nhung node nay anh huong status) ---")

    print("\n--- Nap SS-TRAIN + fit LR_biv / LR_multi / nhu cau phuc vu S_s ---")
    with contextlib.redirect_stdout(buf):
        df_multi = load_multi_service_data(SS_TRAIN, system_type='sockshop', services=SERVICES)
    models_biv, models_multi, demand, fit_meta = fit_cac_bien_the(agent, df_multi, targets)
    print(f"  [OK] {len(df_multi)} dong. Nhu cau phuc vu S_s (CPU/request) uoc luong qua goc:")
    for (svc, mk, suf, col) in targets:
        if mk == 'cpu':
            print(f"    S[{svc}] = {demand[(svc, mk)]:.6f}   (dai fit workload "
                  f"[{fit_meta[(svc, mk)]['biv_min']:.1f}, {fit_meta[(svc, mk)]['biv_max']:.1f}])")

    print("\n--- Du bao tai 288 diem, qua DUNG compute_headroom/summarize_headroom cua He A ---")
    with contextlib.redirect_stdout(buf):
        B, dem = du_bao_tai_cac_diem(agent, cell_df, A, targets, models_biv, models_multi, demand, fit_meta)

    M = A.merge(B, on=['cell', 'muc_tai'], how='inner')
    assert len(M) == 288, f"ghep mat diem: {len(M)}"
    M['A1_kha_thi_long'], M['A1_kha_thi_chat'] = M['heA_on_kha_thi_long'], M['heA_on_kha_thi_chat']
    M['A_off_kha_thi_long'], M['A_off_kha_thi_chat'] = M['heA_off_kha_thi_long'], M['heA_off_kha_thi_chat']
    M['A1_status'], M['A_off_status'] = M['heA_on_status'], M['heA_off_status']
    M['nhom_fit'] = M['cell'].map(gbm.gan_nhom_trong_ngoai)

    # tai su dung bo tong hop/McNemar cua gbm bang cach gan lai 2 hang danh sach cau hinh
    gbm.ALL_CONFIGS = ALL_CONFIGS
    gbm.GB_CONFIGS = BASE_CONFIGS
    summary = gbm.xay_bang_tong_hop(M)
    mcnemar = gbm.mcnemar_tung_cau_hinh(M)
    adm = kiem_admissibility(M, dem, targets, fit_meta, agent)

    M.to_csv(os.path.join(OUT_DIR, 'linear_utillaw_vs_heA_verdict_matrix.csv'), index=False)
    summary.to_csv(os.path.join(OUT_DIR, 'linear_utillaw_vs_heA_summary.csv'), index=False)
    mcnemar.to_csv(os.path.join(OUT_DIR, 'linear_utillaw_vs_heA_mcnemar.csv'), index=False)
    adm.to_csv(os.path.join(OUT_DIR, 'linear_utillaw_admissibility.csv'), index=False)
    print(f"  [OK] da ghi 4 CSV vao {OUT_DIR}")

    print(f"\n{'='*112}\n  PHEP KIEM ADMISSIBILITY -- cau hinh nao co THE bao bao hoa?\n{'='*112}")
    print(adm.to_string(index=False))

    print(f"\n{'='*112}\n  BANG TONG HOP -- KHA THI GIA | BO THIEU\n{'='*112}")
    for stratum in ('tat_ca', 'trong_fit', 'ngoai_fit'):
        print(f"\n  -- stratum: {stratum} --")
        for doc in ('long', 'chat'):
            print(f"   cach doc '{doc}':")
            print(f"   {'cau hinh':14s} {'n':>4s} {'dung':>7s} {'KHA THI GIA':>22s} {'BO THIEU':>22s}")
            sub = summary[(summary.stratum == stratum) & (summary.cach_doc == doc)]
            for cfg in ALL_CONFIGS:
                r = sub[sub.cau_hinh == cfg]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"   {cfg:14s} {int(r.n):4d} {r.dung_pct:6.1f}% "
                      f"{int(r.fp)}/{int(r.n_kt)}={r.fp_pct:5.1f}% [{r.fp_ci[0]:4.1f},{r.fp_ci[1]:4.1f}]%  "
                      f"{int(r.fn)}/{int(r.n_that)}={r.fn_pct:5.1f}% [{r.fn_ci[0]:4.1f},{r.fn_ci[1]:4.1f}]%")

    print(f"\n{'='*112}\n  MCNEMAR vs A1 (ghep doi) -- b = baseline dung & A1 sai, c = baseline sai & A1 dung\n{'='*112}")
    print(f"  {'cau hinh':14s} {'doc':6s} {'stratum':10s} {'n':>4s} {'b':>4s} {'c':>4s} {'p':>8s}  thang A1?")
    for _, r in mcnemar.iterrows():
        print(f"  {r.gb_cau_hinh:14s} {r.cach_doc:6s} {r.stratum:10s} {int(r.n):4d} "
              f"{int(r.b_gb_dung_A1_sai):4d} {int(r.c_gb_sai_A1_dung):4d} {r.p_value:8.4f}  "
              f"{'CO' if r.gb_thang_A1_ve_so_diem else 'khong'}")

    print(f"\n{'='*112}\n  HET.\n{'='*112}")


if __name__ == '__main__':
    main()
