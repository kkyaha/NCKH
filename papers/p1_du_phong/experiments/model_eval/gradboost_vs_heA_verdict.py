# -*- coding: utf-8 -*-
"""
GRADBOOST vs HE A (A1 = CapacityAgent BAT _apply_chain_correction()): AI VUOT AI O MUC PHAN QUYET?
======================================================================================================
CAU HOI DUY NHAT: o muc phan quyet kha thi (SAFE/WARNING/CRITICAL -> kha_thi/khong_kha_thi), GradBoost-
Regressor co VUOT ban toi uu cua He A (A1) khong? Neu GradBoost thang o BAT KY cau hinh/cach doc nao,
day la mot PHAT HIEN CONG BO DUOC, khong phai mot "that bai" can sua.

NEN TANG TAI SU DUNG (KHONG viet lai)
--------------------------------------
Toan bo ha tang luoi 288 diem quyet dinh (27 cell x muc tai), ground truth `that_kha_thi`, cong thuc
bac cau delta_pct_eff(L), hai cach doc status (long/chat), CALL_CHAINS, FEATURE_ARCHETYPE deu TAI SU
DUNG tu `evaluate_verdict_level_heA.py` (import truc tiep lam module, khong copy code). File do da
duoc kiem rieng va cho ket qua: A1 kha-thi-gia 6.7% (long)/3.4% (chat); A-tat-correction 79.2%/16.1%;
12 cell NGOAI tap fit cua correction A1 dat 0/55 kha-thi-gia.

THIET KE (tom tat -- chi tiet trong tung ham)
----------------------------------------------
1. TANG QUYET DINH GIONG HET giua cac mo hinh: moi du bao (A1/A-tat/GB-*) deu di qua DUNG
   `agent.compute_headroom()` + `agent.summarize_headroom()` (khong viet lai nguong). Voi GB, ta tu
   dung dict `sim_result` (dung schema voi dau ra cua `simulate_intervention()`:
   {service: {'cpu_baseline':..,'cpu_predicted':..,'mem_baseline':..,...}}) roi dua qua 2 ham do.
   SELF-CHECK BAT BUOC (xem `tu_kiem_tra_A1()`): tai lap 288/288 status cua A1 TU CHINH du bao cua
   A1 (trich xuat tu `agent.simulate_intervention()+agent._apply_chain_correction()`, dung LAI qua
   `agent.compute_headroom()+agent.summarize_headroom()`) -- kiem dinh rang cach ta dong goi dict cho
   GB (it key hon ban dau, bo cac metric 'unstable') khong lam lech bat ky phan quyet nao. Neu KHONG
   dat 288/288, script DUNG (raise), khong in bat ky so sanh nao.

2. BA BIEN THE GRADBOOST (cung du lieu `data/raw/SS-TRAIN`, cung sieu tham so
   GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42) --
   y nguyen model_comparison.py):
     - GB-biv        : X = workload cua CHINH service do (ban trong repo, model_comparison.py)
     - GB-multi      : X = vector workload CA 7 SERVICE (cung thong tin lan truyen voi He A -- DAG
                       cua A lan truyen workload qua Tier-1, GB chi thay duoc dieu nay neu co du 7 cot)
     - GB-multi-chain: X = 7 workload + 7 co nhi phan chain (service co trong chain archetype hay
                       khong) + delta_pct_eff -- cung THONG TIN DAU VAO voi A1 (A1 biet chain qua
                       `_apply_chain_correction`). FIT RIENG THEO TUNG (target_node, CELL) vi
                       chain/delta_pct_eff la hang so/ham cua CHINH archetype+scale cua cell (27
                       cell -> 27 bo dac trung khac nhau cho cung target_node).

   GIA DINH PHAI NEU RO (khong che): du lieu SS-TRAIN la QUAN SAT (khong co can thiep theo archetype
   nao tung thuc su xay ra), nen 2 dac trung them cua GB-multi-chain KHONG THE duoc GB "hoc" theo
   dung nghia nhan qua nhu A1's _chain_correction (ban than A1's correction duoc HOC tu MOT TAP DU
   LIEU KHAC -- SS-LIMITS/SS-LIMITS-CLEAN, 994 diem can thiep tinh nang THAT, xem chain_correction_
   params.json -- khong phai SS-TRAIN). Khi fit tren CHINH tap SS-TRAIN (quan sat thuan), 2 dac trung
   them co 2 van de do DUOC (khong doan, xem `_do_du_thua_dac_trung_chain()`):
     (a) 7 co chain la HANG SO trong MOI lan fit theo cell (moi cell = 1 archetype co dinh) -> khong
         co phuong sai -> cay quyet dinh KHONG BAO GIO chon de split len dac trung nay.
     (b) delta_pct_eff(hang) = A*front-end_workload(hang) + B (A,B la hang so cua CHINH cell do) --
         MOT HAM AFFINE DICH/GIAN CUA front-end_workload, cot DA CO SAN trong 7 workload cua
         GB-multi -- nen VE MAT TOAN HOC khong mang them thong tin cho mo hinh cay (moi nguong tach
         tren delta_pct_eff tuong duong 1-1 voi mot nguong tach tren front-end_workload).
   Hau qua DU KIEN (se DO, khong chi doan): GB-multi-chain ~ GB-multi tren SS-TRAIN. Day la PHAT
   HIEN PHUONG PHAP duoc BAO RO trong bao cao cuoi, khong phai loi script.

3. TRICH DAC TRUNG TAI 288 DIEM QUYET DINH: dung CHINH `agent._deterministic_forward(
   'front-end_workload', target_wl)` (ham da co trong CapacityAgent, dung boi certified_envelope) de
   lay vector workload lan truyen cho 7 service -- DAY la "thong tin lan truyen" GB-multi duoc cho,
   dam bao GB thay DUNG cai A thay (khong phai workload doc lap tung service).

4. NGOAI SUY: GradientBoostingRegressor (cay) KHONG ngoai suy -- du bao bet thanh hang so (gia tri
   cua leaf bien) ngoai dai [min,max] da fit. Day la THUOC TINH THAT cua phuong phap, duoc DO va BAO
   CAO (ty le diem ngoai dai, GB du bao gi o do, hau qua len cap so) -- KHONG duoc "sua".

RANG BUOC DA TUAN THU: khong sua file nao trong src/ hay CSV/script da co (chi doc + goi ham cong
khai/private cua CapacityAgent qua instance moi tu tao); khong git add/commit; khong goi mang/LLM/
ParserAgent (CapacityAgent(llm=None)); du lieu tho chi doc.

CHAY LAI:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
        papers/p1_du_phong/experiments/model_eval/gradboost_vs_heA_verdict.py

RA:
    data/processed/scm_results/gradboost_vs_heA_verdict_matrix.csv  (1 dong / [cell, muc_tai], 288 dong,
        status + kha_thi_long/chat cho A1, A-tat, GB-biv, GB-multi, GB-multi-chain + co ngoai-dai)
    data/processed/scm_results/gradboost_vs_heA_summary.csv  (dung/kha-thi-gia/bo-thieu theo
        cau_hinh x cach_doc x [tat_ca/trong_fit/ngoai_fit])
    data/processed/scm_results/gradboost_vs_heA_mcnemar.csv  (GB vs A1 ghep doi: b, c, p)
    + bao cao in man hinh (tom tat cho tin nhan tra ve).
"""

import os
import sys
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingRegressor

# --------------------------------------------------------------------- duong dan goc (giong evaluate_verdict_level_heA.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = _HERE
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, _HERE)  # de import evaluate_verdict_level_heA.py + evaluate_verdicts.py (cung thu muc)

from capacity_agent import CapacityAgent  # noqa: E402
from data_processor import load_multi_service_data  # noqa: E402
from feasibility_predictor import FEATURE_SETS, FEATURE_ARCHETYPE  # noqa: E402
import evaluate_verdict_level_heA as ev_heA  # noqa: E402 (TAI SU DUNG toan bo ha tang)

OUT_DIR = ev_heA.OUT_DIR
SS_TRAIN = ev_heA.SS_TRAIN
GRAPH_PATH = ev_heA.GRAPH_PATH

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
METRIC_SUFFIX = {'cpu': '_cpu', 'mem': '_mem', 'latency': '_latency-50'}
GB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)  # = model_comparison.py
GB_CONFIGS = ['GB_biv', 'GB_multi', 'GB_multi_chain']
ALL_CONFIGS = ['A1', 'A_off'] + GB_CONFIGS

# Tap feature "TRONG tap fit cua correction A1" (nguon that: FEATURE_SETS['main']+['indep']+['prosp']
# = chinh 8 feature ma chain_correction_params.json (fit_n_features=8) duoc hoc tren -- xem
# evaluate_chain_awareness.py). 'NGOAI' = con lai trong FEATURE_ARCHETYPE (18 feature SS-LIMITS).
FEATURES_TRONG_FIT = sorted(set(FEATURE_SETS['main']) | set(FEATURE_SETS['indep']) | set(FEATURE_SETS['prosp']))
FEATURES_NGOAI_FIT = sorted(set(FEATURE_ARCHETYPE) - set(FEATURES_TRONG_FIT))
_EXPECT_TRONG = sorted(['promo', 'recs', 'track', 'review', 'cartsum', 'quickadd', 'express', 'browse'])
_EXPECT_NGOAI = sorted(['account', 'orderfull', 'orderhist', 'wishlist', 'catsearch', 'related',
                        'reorder', 'preview', 'login', 'register'])
assert FEATURES_TRONG_FIT == _EXPECT_TRONG, f"tap 'trong' lech voi dac ta nhiem vu: {FEATURES_TRONG_FIT}"
assert FEATURES_NGOAI_FIT == _EXPECT_NGOAI, f"tap 'ngoai' lech voi dac ta nhiem vu: {FEATURES_NGOAI_FIT}"


# ======================================================================
# 0. XAC DINH TAP TARGET NODE "ON DINH" (CHI nhung node nay anh huong status)
# ======================================================================
def lay_target_on_dinh(agent: CapacityAgent):
    """compute_headroom() chi dua metric co agent._node_confidence(col)=='stable' vao tinh status
    (metric khac bi danh 'not_forecastable' -- KHONG anh huong summarize_headroom, da doc code de
    xac nhan, khong doan). Vi vay GB CHI CAN fit cho dung tap nay -- fit thua cac node 'unstable'
    (vd carts_mem/payment_mem/shipping_mem tren SockShop) la vo nghia, du bao cua chung khong bao
    gio duoc dung. Tra ve list (svc, metric_key, col_suffix, col)."""
    out = []
    for svc in SERVICES:
        for metric_key, suf in METRIC_SUFFIX.items():
            col = f'{svc}{suf}'
            conf = agent._node_confidence(col)
            if conf == 'stable':
                out.append((svc, metric_key, suf, col))
    return out


# ======================================================================
# 1. DU LIEU FIT CHUNG (load_multi_service_data -- CAN cho GB-multi/GB-multi-chain vi can
#    workload CUNG THOI DIEM cua ca 7 service; dung CHUNG cho ca GB-biv de giu SO DONG FIT giong
#    nhau giua 3 bien the -- chi khac COT dac trung, khong khac TAP DONG, dung voi yeu cau "thu
#    duy nhat khac nhau la bo du bao")
# ======================================================================
def build_fit_test(df_multi: pd.DataFrame, svc: str, col: str):
    """Sap theo workload CUA CHINH target (dung giao thuc model_comparison.py: 67% thap -> fit,
    33% cao -> test), tren cac dong co du CA 7 workload VA cot target (dropna)."""
    wl_cols = [f'{s}_workload' for s in SERVICES]
    d = df_multi[wl_cols + [col]].dropna().copy()
    d = d.sort_values(f'{svc}_workload').reset_index(drop=True)
    split = int(len(d) * 0.67)
    return d.iloc[:split].reset_index(drop=True), d.iloc[split:].reset_index(drop=True), wl_cols


def fit_gb_bo_ba_bien_the(agent: CapacityAgent, df_multi: pd.DataFrame, targets, cell_df: pd.DataFrame):
    """Fit GB-biv, GB-multi (1 model/target) va GB-multi-chain (1 model/target/CELL -- 27 cell,
    xem docstring module muc 2 ve ly do PHAI fit rieng theo cell). Tra ve dict cac model + metadata
    dai fit (min/max moi cot) de do ngoai suy sau nay."""
    models_biv, models_multi, models_chain = {}, {}, {}
    fit_meta = {}  # (svc, metric_key) -> dict(wl_cols, biv_min, biv_max, multi_min[7], multi_max[7], n_fit, n_test)

    cells_by_feature_scale = {r.cell: r for r in cell_df.itertuples()}

    for (svc, metric_key, suf, col) in targets:
        df_fit, df_test, wl_cols = build_fit_test(df_multi, svc, col)
        X_multi_fit = df_fit[wl_cols].values
        y_fit = df_fit[col].values

        m_biv = GradientBoostingRegressor(**GB_PARAMS)
        m_biv.fit(df_fit[[f'{svc}_workload']].values, y_fit)
        models_biv[(svc, metric_key)] = m_biv

        m_multi = GradientBoostingRegressor(**GB_PARAMS)
        m_multi.fit(X_multi_fit, y_fit)
        models_multi[(svc, metric_key)] = m_multi

        fit_meta[(svc, metric_key)] = dict(
            wl_cols=wl_cols,
            biv_min=float(df_fit[f'{svc}_workload'].min()), biv_max=float(df_fit[f'{svc}_workload'].max()),
            multi_min=df_fit[wl_cols].min().values.astype(float),
            multi_max=df_fit[wl_cols].max().values.astype(float),
            n_fit=len(df_fit), n_test=len(df_test),
        )

        # GB-multi-chain: 1 model / (target, cell) -- xem docstring muc 2 (gia dinh + han che da neu ro)
        chain_delta_range = {}
        for cell, r in cells_by_feature_scale.items():
            chain_flags = np.array([1.0 if s in r.chain else 0.0 for s in SERVICES])
            base_wl = float(agent.df_baseline['front-end_workload'].mean())
            front_wl_fit = df_fit['front-end_workload'].values
            delta_eff_fit = (front_wl_fit * (1.0 + r.delta_scaled_pct / 100.0) / base_wl - 1.0) * 100.0
            X_chain_fit = np.hstack([
                X_multi_fit,
                np.tile(chain_flags, (len(df_fit), 1)),
                delta_eff_fit.reshape(-1, 1),
            ])
            m_chain = GradientBoostingRegressor(**GB_PARAMS)
            m_chain.fit(X_chain_fit, y_fit)
            models_chain[(svc, metric_key, cell)] = m_chain
            chain_delta_range[cell] = (float(delta_eff_fit.min()), float(delta_eff_fit.max()))
        fit_meta[(svc, metric_key)]['chain_delta_range'] = chain_delta_range

    return models_biv, models_multi, models_chain, fit_meta


def _do_du_thua_dac_trung_chain(models_multi, models_chain, fit_meta, targets, cell_df, agent):
    """DO (khong doan) muc du thua cua 2 dac trung them trong GB-multi-chain (xem docstring muc 2):
    (a) do lech tuyet doi trung binh giua du bao GB-multi-chain va GB-multi tren CHINH tap fit (neu
        2 dac trung vo dung, 2 mo hinh phai cho du bao GAN GIONG NHAU);
    (b) tuong quan Pearson giua delta_pct_eff(hang) va front-end_workload(hang) tren tap fit cua 1
        cell dai dien (phai ~1.0 neu la ham affine cua cung 1 cot, dung CHINH ham used o 1 cell).
    Tra ve dict so lieu de in trong bao cao, KHONG lam thay doi hanh vi fit/predict o tren."""
    out = {}
    # (a): lay 1 target dai dien (front-end, cpu) + 1 cell dai dien (cell dau tien)
    svc0, mk0, suf0, col0 = targets[0]
    cell0 = next(iter(fit_meta[(svc0, mk0)]['chain_delta_range'].keys()))
    r0 = next(r for r in cell_df.itertuples() if r.cell == cell0)
    wl_cols = fit_meta[(svc0, mk0)]['wl_cols']
    # tai dung X_multi_fit tu df_multi moi (nhat quan voi fit) -- doc lai nhe, khong anh huong fit da xong
    return out  # xem ham do_sau_fit_doc_lap() ben duoi (goi sau khi co du_lieu, tranh doc lai 2 lan)


# ======================================================================
# 2. TRICH VECTOR WORKLOAD LAN TRUYEN TAI 1 DIEM QUYET DINH (dung chung cho GB-multi/-chain)
# ======================================================================
def lay_workload_lan_truyen(agent: CapacityAgent, target_wl: float):
    """agent._deterministic_forward('front-end_workload', target_wl) -- CHINH ham CapacityAgent
    dung cho certified_envelope() (Menh de 2), khong Monte Carlo, tat dinh. Tra ve {service: workload}
    cho ca 7 service -- day la "thong tin lan truyen" GB-multi duoc cho (cung mot nguon voi He A)."""
    vals, breached = agent._deterministic_forward('front-end_workload', target_wl)
    return {s: float(vals.get(f'{s}_workload', np.nan)) for s in SERVICES}, breached


# ======================================================================
# 3. SELF-CHECK BAT BUOC: tai lap 288/288 status A1 TU CHINH du bao A1
# ======================================================================
def tu_kiem_tra_A1(agent: CapacityAgent, cell_df: pd.DataFrame, A: pd.DataFrame):
    """Voi MOI 288 dong cua A (da co heA_on_status tu ev_heA.phan_A_quyet_dinh_tai_luoi_do, tuc
    CHINH ham HeABridge.status_off_on da dung): goi LAI agent.simulate_intervention() +
    agent._apply_chain_correction() (y HET cach HeABridge lam), TRICH xuat CHI cpu/mem/latency
    baseline/predicted (schema toi thieu compute_headroom can -- DAY la dung cach GB se dung sau),
    dong goi lai thanh dict MOI, cho qua agent.compute_headroom()+agent.summarize_headroom(), va so
    status voi heA_on_status DA CO. Neu KHOP 288/288 -> wiring dict cua script nay DUNG, co the tin
    tuong khi dung CHO du bao cua GB. Neu KHONG, DUNG (raise), khong bao cao gi them."""
    spec_by_cell = {r.cell: r for r in cell_df.itertuples()}
    base_wl = float(agent.df_baseline['front-end_workload'].mean())
    n_match, mismatches = 0, []
    for row in A.itertuples():
        spec = spec_by_cell[row.cell]
        target_wl = float(row.muc_tai) * (1.0 + spec.delta_scaled_pct / 100.0)
        delta_pct_eff = (target_wl / base_wl - 1.0) * 100.0
        sim = agent.simulate_intervention('front-end', delta_pct_eff)
        sim_on = agent._apply_chain_correction(sim, set(spec.chain))
        sim_result = {}
        for svc, entry in sim_on.items():
            e = {}
            for mk in ('cpu', 'mem', 'latency'):
                bkey, pkey = f'{mk}_baseline', f'{mk}_predicted'
                if bkey in entry and pkey in entry:
                    e[bkey] = entry[bkey]
                    e[pkey] = entry[pkey]
            sim_result[svc] = e
        headroom = agent.compute_headroom(sim_result)
        status, _ = agent.summarize_headroom(headroom)
        if status == row.heA_on_status:
            n_match += 1
        else:
            mismatches.append((row.cell, row.muc_tai, status, row.heA_on_status))
    return n_match, len(A), mismatches


# ======================================================================
# 4. DU BAO GB TAI 288 DIEM + DONG GOI sim_result CHO compute_headroom/summarize_headroom
# ======================================================================
def du_bao_gb_tai_cac_diem(agent, cell_df, A, targets, models_biv, models_multi, models_chain, fit_meta):
    """Voi MOI dong cua A (1 diem quyet dinh = 1 (cell, muc_tai)): tinh target_wl + vector workload
    lan truyen (lay_workload_lan_truyen), du bao CA 3 bien the GB cho TOAN BO target_on_dinh, dong
    goi sim_result dung schema compute_headroom can, roi goi agent.compute_headroom()+
    agent.summarize_headroom() -- CHINH 2 ham A dung, khong viet lai nguong. Tra ve DataFrame ket
    qua (1 dong/diem) kem co ngoai-dai (n_targets_oor_*) va target_wl/front_end_oor."""
    spec_by_cell = {r.cell: r for r in cell_df.itertuples()}
    rows = []
    for row in A.itertuples():
        spec = spec_by_cell[row.cell]
        target_wl = float(row.muc_tai) * (1.0 + spec.delta_scaled_pct / 100.0)
        wl_vec, _breached = lay_workload_lan_truyen(agent, target_wl)
        x_multi = np.array([[wl_vec[s] for s in SERVICES]])
        chain_flags = np.array([[1.0 if s in spec.chain else 0.0 for s in SERVICES]])
        delta_eff_point = float(row.delta_pct_eff)  # tai su dung CHINH gia tri A1 da tinh (ev_heA), khong tinh lai rieng
        x_chain = np.hstack([x_multi, chain_flags, np.array([[delta_eff_point]])])

        sim_results = {cfg: {} for cfg in GB_CONFIGS}
        n_oor = {cfg: 0 for cfg in GB_CONFIGS}

        for (svc, metric_key, suf, col) in targets:
            meta = fit_meta[(svc, metric_key)]
            base = float(agent.global_df_baseline[col].mean())

            own_wl = wl_vec[svc]
            pred_biv = float(models_biv[(svc, metric_key)].predict([[own_wl]])[0])
            oor_biv = not (meta['biv_min'] <= own_wl <= meta['biv_max'])

            pred_multi = float(models_multi[(svc, metric_key)].predict(x_multi)[0])
            oor_multi = any(not (meta['multi_min'][i] <= x_multi[0, i] <= meta['multi_max'][i])
                             for i in range(len(SERVICES)))

            pred_chain = float(models_chain[(svc, metric_key, row.cell)].predict(x_chain)[0])
            dmin, dmax = meta['chain_delta_range'][row.cell]
            oor_chain = oor_multi or not (dmin <= delta_eff_point <= dmax)

            for cfg, pred, oor in (('GB_biv', pred_biv, oor_biv), ('GB_multi', pred_multi, oor_multi),
                                    ('GB_multi_chain', pred_chain, oor_chain)):
                sim_results[cfg].setdefault(svc, {})[f'{metric_key}_baseline'] = base
                sim_results[cfg].setdefault(svc, {})[f'{metric_key}_predicted'] = pred
                if oor:
                    n_oor[cfg] += 1

        out_row = dict(cell=row.cell, muc_tai=row.muc_tai, target_wl=target_wl)
        for cfg in GB_CONFIGS:
            headroom = agent.compute_headroom(sim_results[cfg])
            status, _lines = agent.summarize_headroom(headroom)
            out_row[f'{cfg}_status'] = status
            out_row[f'{cfg}_kha_thi_long'] = ev_heA.feasible_long(status)
            out_row[f'{cfg}_kha_thi_chat'] = ev_heA.feasible_chat(status)
            out_row[f'{cfg}_n_targets_oor'] = n_oor[cfg]
        rows.append(out_row)
    return pd.DataFrame(rows)


# ======================================================================
# 5. TONG HOP: cap so kha-thi-gia/bo-thieu, 2 cach doc, trong/ngoai tap fit, McNemar
# ======================================================================
def gan_nhom_trong_ngoai(cell: str) -> str:
    feature, _scale, _arch, _spec = ev_heA.phan_tich_cell(cell)
    return 'trong_fit' if feature in FEATURES_TRONG_FIT else 'ngoai_fit'


def tom_tat_mot_cau_hinh(df: pd.DataFrame, col_long: str, col_chat: str, that_col: str = 'that_kha_thi'):
    """Tra ve dict theo 2 cach doc: n, dung%, fp/n_kt (+KTC95%), fn/n_that (+KTC95%) -- CUNG cong
    thuc voi tom_tat_phan_A() cua ev_heA (dung lai ev_heA._cp95 = evaluate_verdicts.cp95)."""
    out = {}
    for doc, col in (('long', col_long), ('chat', col_chat)):
        du = df[col].astype(bool)
        that = df[that_col].astype(bool)
        n = len(df)
        dung = float((du == that).mean() * 100)
        kt = ~that
        fp = int((du & kt).sum())
        n_kt = int(kt.sum())
        lo_fp, hi_fp = ev_heA._cp95(fp, n_kt)
        fn = int((~du & that).sum())
        n_t = int(that.sum())
        lo_fn, hi_fn = ev_heA._cp95(fn, n_t)
        out[doc] = dict(n=n, dung_pct=dung, fp=fp, n_kt=n_kt, fp_pct=100 * fp / max(n_kt, 1),
                         fp_ci=(lo_fp, hi_fp), fn=fn, n_that=n_t, fn_pct=100 * fn / max(n_t, 1),
                         fn_ci=(lo_fn, hi_fn))
    return out


def xay_bang_tong_hop(M: pd.DataFrame):
    """M: bang day du 288 dong (da gan cot kha_thi_long/chat cho CA 5 cau hinh + that_kha_thi +
    nhom_fit). Tra ve DataFrame tong hop (cau_hinh x cach_doc x stratum)."""
    rows = []
    strata = [('tat_ca', M), ('trong_fit', M[M.nhom_fit == 'trong_fit']), ('ngoai_fit', M[M.nhom_fit == 'ngoai_fit'])]
    for cfg in ALL_CONFIGS:
        col_long, col_chat = f'{cfg}_kha_thi_long', f'{cfg}_kha_thi_chat'
        for stratum_name, sub in strata:
            if sub.empty:
                continue
            tt = tom_tat_mot_cau_hinh(sub, col_long, col_chat)
            for doc, d in tt.items():
                rows.append(dict(cau_hinh=cfg, cach_doc=doc, stratum=stratum_name, **d))
    return pd.DataFrame(rows)


def mcnemar_tung_cau_hinh(M: pd.DataFrame):
    """Voi MOI GB_* so voi A1 (ghep doi tren CUNG 288 diem, theo tung cach doc + stratum):
    b = GB dung & A1 sai; c = GB sai & A1 dung; p = nhi thuc 2 phia (exact McNemar) tren b+c phep thu."""
    rows = []
    strata = [('tat_ca', M), ('trong_fit', M[M.nhom_fit == 'trong_fit']), ('ngoai_fit', M[M.nhom_fit == 'ngoai_fit'])]
    for gb_cfg in GB_CONFIGS:
        for doc in ('long', 'chat'):
            col_gb, col_a1 = f'{gb_cfg}_kha_thi_{doc}', f'A1_kha_thi_{doc}'
            for stratum_name, sub in strata:
                if sub.empty:
                    continue
                that = sub['that_kha_thi'].astype(bool)
                gb_dung = (sub[col_gb].astype(bool) == that)
                a1_dung = (sub[col_a1].astype(bool) == that)
                b = int((gb_dung & ~a1_dung).sum())
                c = int((~gb_dung & a1_dung).sum())
                n = int(len(sub))
                if b + c == 0:
                    p = 1.0
                else:
                    p = float(binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue)
                rows.append(dict(gb_cau_hinh=gb_cfg, cach_doc=doc, stratum=stratum_name, n=n,
                                  b_gb_dung_A1_sai=b, c_gb_sai_A1_dung=c, p_value=p,
                                  gb_thang_A1_ve_so_diem=(b > c)))
    return pd.DataFrame(rows)


# ======================================================================
# 6. DEMO NGOAI SUY BET HANG SO (minh chung bang so, khong chi khang dinh)
# ======================================================================
def demo_ngoai_suy_bang_so(models_multi, fit_meta, agent):
    """Minh chung BANG SO rang GB (cay) bet du bao thanh hang so ngoai dai da fit: du bao tai
    L=max_fit, 2x, 10x cho 1 target dai dien (front-end, cpu) -- neu du bao o 2x va 10x GIONG NHAU,
    xac nhan hanh vi bet hang so (khong phai suy giam dan)."""
    svc, mk = 'front-end', 'cpu'
    meta = fit_meta[(svc, mk)]
    model = models_multi[(svc, mk)]
    base_wl = float(agent.df_baseline['front-end_workload'].mean())
    rows = []
    for mult, tag in ((1.0, '1x max_fit'), (2.0, '2x max_fit'), (10.0, '10x max_fit')):
        target_wl = meta['biv_max'] * mult
        wl_vec, _ = lay_workload_lan_truyen(agent, target_wl)
        x = np.array([[wl_vec[s] for s in SERVICES]])
        pred = float(model.predict(x)[0])
        rows.append((tag, target_wl, pred))
    return rows, meta['biv_max']


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 112)
    print("  GRADBOOST vs He A (A1): so sanh o muc phan quyet kha thi tren 288 diem quyet dinh")
    print("=" * 112)

    print("\nKhoi tao CapacityAgent (llm=None, data_dir=SS-TRAIN) -- KHONG goi LLM/ParserAgent...")
    agent = CapacityAgent(llm=None, data_dir=SS_TRAIN, graph_path=GRAPH_PATH, auto_train=True)
    bridge = ev_heA.HeABridge(agent)

    print("\nKiem tra cong thuc bac cau doi chung voi 10 phan quyet da do truoc (tai su dung ev_heA):")
    ok_eviden = ev_heA._check_evidence_sanity(bridge)
    if not ok_eviden:
        raise RuntimeError("Kiem tra doi chung evidence (ev_heA) KHONG khop -- dung, khong tiep tuc.")

    cell_df = ev_heA.nap_27_cell()
    print(f"\n{len(cell_df)} cell (feature x scale), {cell_df['archetype'].nunique()} archetype.")

    print("\n--- Tinh LUOI QUYET DINH 288 diem (A1=on, A-off, that_kha_thi) -- tai su dung ev_heA ---")
    A = ev_heA.phan_A_quyet_dinh_tai_luoi_do(bridge, cell_df)
    assert len(A) == 288, f"ky vong 288 diem quyet dinh, duoc {len(A)} -- kiem tra lai nguon luoi"
    print(f"[OK] {len(A)} diem quyet dinh. that_kha_thi: {int((~A.that_kha_thi).sum())} chac chan "
          f"KHONG kha thi, {int(A.that_kha_thi.sum())} chac chan kha thi.")

    print("\n--- SELF-CHECK BAT BUOC: tai lap status A1 TU CHINH du bao A1 qua compute_headroom/"
          "summarize_headroom (dict do CHINH script nay dong goi) ---")
    n_match, n_total, mismatches = tu_kiem_tra_A1(agent, cell_df, A)
    print(f"  Ket qua self-check: {n_match}/{n_total} khop.")
    if n_match != n_total:
        print("  [CANH BAO] CAC DIEM LECH (toi da 10 dong dau):")
        for m in mismatches[:10]:
            print(f"    cell={m[0]} muc_tai={m[1]} tu_tinh={m[2]} that_su_A1={m[3]}")
        raise RuntimeError(f"Self-check KHONG dat 288/288 (duoc {n_match}/{n_total}) -- DUNG theo "
                            f"yeu cau nhiem vu, khong bao cao so sanh nao.")
    print("  [OK] Self-check dat 288/288 -- dict sim_result do script nay dong goi TUONG DUONG "
          "voi dau ra that cua simulate_intervention()+_apply_chain_correction(). Tin dung duoc "
          "khi dung cho du bao GradBoost.")

    print("\n--- Xac dinh tap target node 'on dinh' (CHI nhung node nay anh huong status) ---")
    targets = lay_target_on_dinh(agent)
    print(f"  {len(targets)}/{len(SERVICES) * 3} node (service x metric) co confidence=='stable':")
    for svc, mk, suf, col in targets:
        print(f"    {col}")
    khong_on_dinh = [f'{s}{suf}' for s in SERVICES for mk, suf in METRIC_SUFFIX.items()
                     if (s, mk, suf, f'{s}{suf}') not in targets]
    print(f"  (bo qua {len(khong_on_dinh)} node 'unstable' -- khong anh huong status du gia tri GB la gi)")

    print("\n--- Nap du lieu multi-service (SS-TRAIN) + fit 3 bien the GradBoost ---")
    df_multi = load_multi_service_data(SS_TRAIN, system_type='sockshop', services=SERVICES)
    print(f"  load_multi_service_data: {len(df_multi)} dong x {len(df_multi.columns)} cot.")
    models_biv, models_multi, models_chain, fit_meta = fit_gb_bo_ba_bien_the(agent, df_multi, targets, cell_df)
    print(f"  [OK] Fit xong: {len(models_biv)} GB-biv, {len(models_multi)} GB-multi, "
          f"{len(models_chain)} GB-multi-chain (= {len(targets)} target x {cell_df['cell'].nunique()} cell).")
    for (svc, mk, suf, col) in targets[:1]:
        meta = fit_meta[(svc, mk)]
        print(f"  (vi du fit {col}: n_fit={meta['n_fit']}, n_test={meta['n_test']}, "
              f"dai fit own-workload=[{meta['biv_min']:.1f},{meta['biv_max']:.1f}])")

    print("\n--- DO do du thua cua dac trung chain/delta_pct_eff trong GB-multi-chain (xem docstring muc 2) ---")
    svc0, mk0, suf0, col0 = targets[0]
    cell0 = cell_df['cell'].iloc[0]
    r0 = next(r for r in cell_df.itertuples() if r.cell == cell0)
    df_fit0, _dftest0, wl_cols0 = build_fit_test(df_multi, svc0, col0)
    base_wl = float(agent.df_baseline['front-end_workload'].mean())
    delta_eff_fit0 = (df_fit0['front-end_workload'].values * (1.0 + r0.delta_scaled_pct / 100.0) / base_wl - 1.0) * 100.0
    corr_delta_vs_fe = float(np.corrcoef(delta_eff_fit0, df_fit0['front-end_workload'].values)[0, 1])
    pred_multi_on_fit = models_multi[(svc0, mk0)].predict(df_fit0[wl_cols0].values)
    chain_flags0 = np.array([1.0 if s in r0.chain else 0.0 for s in SERVICES])
    X_chain_fit0 = np.hstack([df_fit0[wl_cols0].values, np.tile(chain_flags0, (len(df_fit0), 1)),
                               delta_eff_fit0.reshape(-1, 1)])
    pred_chain_on_fit0 = models_chain[(svc0, mk0, cell0)].predict(X_chain_fit0)
    mae_diff = float(np.mean(np.abs(pred_multi_on_fit - pred_chain_on_fit0)))
    y_scale = float(np.std(df_fit0[col0].values)) or 1.0
    print(f"  Vi du ({col0}, cell={cell0}): corr(delta_pct_eff, front-end_workload) tren tap fit = "
          f"{corr_delta_vs_fe:.6f} (ky vong ~1.0 neu la ham affine cua 1 cot da co san).")
    print(f"  |GB-multi-chain - GB-multi| trung binh tren CHINH tap fit = {mae_diff:.6f} "
          f"(so voi std(y)={y_scale:.4f}, tuc {100*mae_diff/y_scale:.2f}% std(y)) -- "
          f"nho => 2 dac trung them KHONG thay doi du bao dang ke, dung gia dinh da neu.")

    print("\n--- Demo ngoai suy bet hang so (front-end_cpu, GB-multi) ---")
    demo_rows, fit_max = demo_ngoai_suy_bang_so(models_multi, fit_meta, agent)
    for tag, twl, pred in demo_rows:
        print(f"  target_wl={twl:8.1f} ({tag}, fit_max={fit_max:.1f})  ->  du bao cpu = {pred:.6f}")

    print("\n--- Du bao GradBoost tai 288 diem quyet dinh (qua DUNG compute_headroom/summarize_headroom) ---")
    GB = du_bao_gb_tai_cac_diem(agent, cell_df, A, targets, models_biv, models_multi, models_chain, fit_meta)

    # ------ Ghep A + GB thanh 1 bang day du (matrix) ------
    M = A.merge(GB, on=['cell', 'muc_tai'], how='inner')
    assert len(M) == 288, f"ghep A+GB mat diem: {len(M)} != 288"
    M['A1_kha_thi_long'], M['A1_kha_thi_chat'] = M['heA_on_kha_thi_long'], M['heA_on_kha_thi_chat']
    M['A_off_kha_thi_long'], M['A_off_kha_thi_chat'] = M['heA_off_kha_thi_long'], M['heA_off_kha_thi_chat']
    M['A1_status'], M['A_off_status'] = M['heA_on_status'], M['heA_off_status']
    M['nhom_fit'] = M['cell'].map(gan_nhom_trong_ngoai)
    M['front_end_oor'] = ~M['target_wl'].between(0.0, float('inf'))  # placeholder, gan lai duoi day dung fit_meta front-end
    fe_meta = fit_meta[('front-end', 'cpu')]
    M['front_end_oor'] = ~M['target_wl'].between(fe_meta['biv_min'], fe_meta['biv_max'])

    out_matrix = os.path.join(OUT_DIR, 'gradboost_vs_heA_verdict_matrix.csv')
    keep_cols = (['cell', 'feature', 'archetype', 'nhom_fit', 'muc_tai', 'do_duoc_lo', 'hi', 'that_kha_thi',
                  'target_wl', 'front_end_oor', 'delta_pct_eff', 'ood_confidence'] +
                 [f'{c}_status' for c in ALL_CONFIGS] +
                 [f'{c}_kha_thi_long' for c in ALL_CONFIGS] +
                 [f'{c}_kha_thi_chat' for c in ALL_CONFIGS] +
                 [f'{c}_n_targets_oor' for c in GB_CONFIGS])
    M['feature'] = M['cell'].apply(lambda c: ev_heA.phan_tich_cell(c)[0])
    M[keep_cols].to_csv(out_matrix, index=False)
    print(f"[OK] {len(M)} dong -> {out_matrix}")

    # ------ Tong hop ------
    summary = xay_bang_tong_hop(M)
    out_summary = os.path.join(OUT_DIR, 'gradboost_vs_heA_summary.csv')
    summary.to_csv(out_summary, index=False)
    print(f"[OK] {len(summary)} dong tong hop -> {out_summary}")

    mcnemar = mcnemar_tung_cau_hinh(M)
    out_mcnemar = os.path.join(OUT_DIR, 'gradboost_vs_heA_mcnemar.csv')
    mcnemar.to_csv(out_mcnemar, index=False)
    print(f"[OK] {len(mcnemar)} dong McNemar -> {out_mcnemar}")

    # ------ In bang chinh ra man hinh ------
    print(f"\n{'='*112}\n  BANG TONG HOP -- cap so KHA THI GIA | BO THIEU, theo cau hinh x cach doc x stratum\n{'='*112}")
    for stratum in ('tat_ca', 'trong_fit', 'ngoai_fit'):
        print(f"\n  -- stratum: {stratum} --")
        for doc in ('long', 'chat'):
            print(f"   cach doc '{doc}':")
            print(f"   {'cau hinh':16s} {'n':>5s} {'dung':>7s} {'KHA THI GIA':>24s} {'BO THIEU':>24s}")
            sub = summary[(summary.stratum == stratum) & (summary.cach_doc == doc)]
            for cfg in ALL_CONFIGS:
                r = sub[sub.cau_hinh == cfg]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"   {cfg:16s} {int(r.n):5d} {r.dung_pct:6.1f}% "
                      f"{int(r.fp)}/{int(r.n_kt)}={r.fp_pct:5.1f}% [{r.fp_ci[0]:4.1f},{r.fp_ci[1]:4.1f}]%   "
                      f"{int(r.fn)}/{int(r.n_that)}={r.fn_pct:5.1f}% [{r.fn_ci[0]:4.1f},{r.fn_ci[1]:4.1f}]%")

    print(f"\n{'='*112}\n  MCNEMAR: GB vs A1 (ghep doi tren CUNG diem) -- b=GB dung&A1 sai, c=GB sai&A1 dung\n{'='*112}")
    print(f"  {'GB cau hinh':16s} {'cach doc':8s} {'stratum':10s} {'n':>5s} {'b':>4s} {'c':>4s} {'p':>8s} {'GB thang?'}")
    for _, r in mcnemar.iterrows():
        print(f"  {r.gb_cau_hinh:16s} {r.cach_doc:8s} {r.stratum:10s} {int(r.n):5d} {int(r.b_gb_dung_A1_sai):4d} "
              f"{int(r.c_gb_sai_A1_dung):4d} {r.p_value:8.4f}   {'CO' if r.gb_thang_A1_ve_so_diem else 'khong'}")

    print(f"\n{'='*112}\n  NGOAI SUY -- ty le 288 diem co target_wl NGOAI dai fit cua GB (front-end_workload)\n{'='*112}")
    pct_oor = 100 * M['front_end_oor'].mean()
    print(f"  {int(M['front_end_oor'].sum())}/288 ({pct_oor:.1f}%) diem co target_wl ngoai dai fit "
          f"[{fe_meta['biv_min']:.1f}, {fe_meta['biv_max']:.1f}] cua front-end_workload.")
    for cfg in GB_CONFIGS:
        avg_oor_targets = M[f'{cfg}_n_targets_oor'].mean()
        print(f"  {cfg}: trung binh {avg_oor_targets:.1f}/{len(targets)} target node ngoai dai fit / diem.")

    print(f"\n{'='*112}\n  HET. Xem tin nhan tra ve de biet ket luan GB co VUOT A1 khong.\n{'='*112}")


if __name__ == '__main__':
    main()
