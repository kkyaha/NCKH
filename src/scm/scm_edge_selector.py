# -*- coding: utf-8 -*-
"""
SCM EDGE SELECTOR TONG QUAT (system-agnostic)
================================================
Tu dong hoa 3 buoc dau trong quy trinh them canh "backpressure" (R->R, vd
CPU cua caller -> CPU cua callee) vao Global Causal DAG cua CapacityAgent --
buoc nay hien dang lap lai THU CONG cho tung he thong rieng (xem
src/agents/capacity_agent.py:402-469): Sock Shop co 3 canh chon tay,
Train Ticket co ~50 canh liet ke tay, moi he dung mot nguong R2-gain khac
nhau, va script chan doan (experiments/call_chain_neighbor_diagnostic.py)
ma cung ca danh sach SERVICES lan EDGES cho Sock Shop.

Module nay khong bo do()/SCM -- nguoc lai, GIU nguyen co che nhan qua
(dowhy.gcm) va TU DONG HOA + LAM CHAT hon buoc chon canh truoc khi do() duoc
dung de du bao, bang 3 pha:

  Pha 1 (candidate_edges_from_graph)  : liet ke canh R->R ung vien THANG tu
      chinh do thi phu thuoc (khong go tay danh sach) -- dung
      taxonomy_builder.load_graph(), nen nhan bat ky graph JSON nao dung
      schema chung (vd do src/graph/extract_graph_generic.py sinh ra).
  Pha 2 (score_r2_gain)               : do R2 tang bao nhieu khi them metric
      cua caller lam parent thu 2 cho callee (tai lap H3 trong
      call_chain_neighbor_diagnostic.py, nhung THAM SO HOA theo services/
      candidate_edges thay vi go cung).
  Pha 3 (select_edges_by_knee_point)  : chon nguong TU DONG bang diem khuyu
      tay (knee point) tren duong gain da sap xep, thay vi mot nguong co
      dinh khac nhau moi he thong (Sock Shop: top-3 tay; Train Ticket:
      gain>0.03 tay) -- MOT quy tac duy nhat cho moi he thong.

Pha 4 (validate_ood_safety) tai lap experiments/backpressure_edge_ood_safety_test.py
o dang ham dung lai duoc: quet do() qua nhieu delta workload, dem so ca
sign-inversion VOI vs KHONG co tung canh ung vien -- day la phan "toi uu
phep do" thuc su: khong con canh nao duoc dua vao do() model chi vi MAPE
held-out giam (bai hoc tu paper, xem docs/paper_draft.tex Section
"The Backpressure Extension: A Cautionary Case"), ma phai qua ca gate an
toan duoi can thiep that.

select_scm_edges() la ham tong hop 4 pha tren thanh MOT loi goi duy nhat,
dung cho bat ky he thong nao co (a) graph JSON dung schema chung va (b) mot
DataFrame telemetry voi cot '<service>_workload' / '<service>_<metric>'.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from taxonomy_builder import load_graph  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def candidate_edges_from_graph(graph_json_path: str, metric: str = 'cpu',
                                available_columns=None) -> list:
    """Tra ve danh sach (caller, callee) tu CHINH canh caller->callee that
    trong do thi phu thuoc, khong go tay.

    available_columns: neu truyen vao (vd df_data.columns), loc bot canh ma
    he thong khong co du lieu cho metric do -- tranh crash o pha do.
    """
    adj, _node_types = load_graph(graph_json_path)
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    if available_columns is not None:
        cols = set(available_columns)
        edges = [
            (u, v) for u, v in edges
            if f'{u}_{metric}' in cols and f'{v}_{metric}' in cols and f'{v}_workload' in cols
        ]
    return edges


def score_r2_gain(df_data: pd.DataFrame, candidate_edges: list, metric: str = 'cpu',
                   min_rows: int = 200) -> pd.DataFrame:
    """Voi moi canh (caller, callee) ung vien: do R2 cua
    callee_metric ~ callee_workload (baseline) so voi
    callee_metric ~ callee_workload + caller_metric (extended).

    Tai lap H3 trong call_chain_neighbor_diagnostic.py nhung khong go cung
    SERVICES/EDGES -- nhan candidate_edges lam tham so, nen dung duoc cho
    bat ky so luong service/canh nao.

    Tra ve DataFrame [caller, callee, r2_baseline, r2_extended, r2_gain],
    sap xep giam dan theo r2_gain.
    """
    rows = []
    for caller, callee in candidate_edges:
        wl_col, target_col, caller_col = f'{callee}_workload', f'{callee}_{metric}', f'{caller}_{metric}'
        needed = [wl_col, target_col, caller_col]
        if not all(c in df_data.columns for c in needed):
            continue
        sub = df_data[needed].dropna()
        if len(sub) < min_rows:
            continue

        m1 = LinearRegression(positive=True).fit(sub[[wl_col]], sub[target_col])
        r2_baseline = m1.score(sub[[wl_col]], sub[target_col])

        m2 = LinearRegression(positive=True).fit(sub[[wl_col, caller_col]], sub[target_col])
        r2_extended = m2.score(sub[[wl_col, caller_col]], sub[target_col])

        rows.append({
            'caller': caller, 'callee': callee,
            'r2_baseline': round(r2_baseline, 4),
            'r2_extended': round(r2_extended, 4),
            'r2_gain': round(r2_extended - r2_baseline, 4),
            'n_rows': len(sub),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values('r2_gain', ascending=False).reset_index(drop=True)


def score_edge_heldout(df_data: pd.DataFrame, target_col: str, split_col: str,
                        baseline_parents: list, extended_parents: list,
                        mechanism_factory=None, min_rows: int = 200,
                        split_ratio: float = 0.67, rho: float = 2.0):
    """Cham diem MOT canh ung vien bang giao thuc HELD-OUT (OOD Gold
    Standard 67/33) thay vi in-sample.

    VI SAO HELD-OUT (thay doi thang 9/2026, thay cho score_r2_gain/
    score_bic_gain in-sample): toan bo epistemics cua du an (RQ1,
    node_impact.evaluate_node_stability, paper Section "Experimental
    Setup") deu danh gia tren tap tai CAO chua tung thay. Chon canh bang
    diem IN-SAMPLE roi chi kiem an toan dau (OOD-safety) la khong nhat
    quan voi dieu do, va da co bang chung truc tiep no chon sai:
      - Danh sach 50 canh CPU cu (chon bang in-sample R2-gain>0.03):
        ts-travel2-service_cpu MAPE held-out 400%, ts-config-service_cpu
        137% -- overfit nang.
      - BIC (CausIL) cung la tieu chi in-sample: tren Sock Shop no chon
        giu canh orders->payment lam MAPE held-out 298% -> 499%.
    Phat phuc tap kieu BIC chi phat SO THAM SO, khong phat duoc viec mo
    hinh sup do khi phan phoi doi -- chi do truc tiep tren tap tai cao
    moi bat duoc.

    split_col: cot dung de sap xep truoc khi cat 67/33 -- luon la workload
    CUA CHINH service so huu target (train tai thap -> test tai cao).
    mechanism_factory(): tra ve estimator CHUA fit, mac dinh
    LinearRegression(positive=True). Nguoi goi nen truyen dung loai model
    ma production se fit cho node do (vd QueueingLatencyRegressor cho
    latency) de diem so phan anh dung thu se duoc dung that.

    Tra ve dict cac chi so, hoac None neu khong du du lieu.
    """
    if mechanism_factory is None:
        def mechanism_factory():
            return LinearRegression(positive=True)

    needed = sorted(set([target_col, split_col] + baseline_parents + extended_parents))
    if not all(c in df_data.columns for c in needed):
        return None
    sub = df_data[needed].dropna()
    if len(sub) < min_rows:
        return None

    sub = sub.sort_values(split_col).reset_index(drop=True)
    cut = int(len(sub) * split_ratio)
    train, test = sub.iloc[:cut], sub.iloc[cut:]
    if len(test) < 10 or len(train) < 10:
        return None

    y_tr, y_te = train[target_col].values, test[target_col].values

    def _fit_score(parents):
        m = mechanism_factory()
        m.fit(train[parents].values, y_tr)
        pred_te = np.asarray(m.predict(test[parents].values), dtype=float)
        pred_tr = np.asarray(m.predict(train[parents].values), dtype=float)
        ss_res = float(np.sum((y_te - pred_te) ** 2))
        ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        mask = y_te != 0
        mape = (float(np.mean(np.abs((y_te[mask] - pred_te[mask]) / y_te[mask])) * 100)
                if mask.sum() > 0 else float('nan'))
        rss_tr = max(float(np.sum((y_tr - pred_tr) ** 2)), 1e-12)
        n_tr = len(y_tr)
        bic = n_tr * np.log(rss_tr / n_tr) + rho * (len(parents) + 1) * np.log(n_tr)
        return r2, mape, bic, m

    r2_b, mape_b, bic_b, _ = _fit_score(baseline_parents)
    r2_e, mape_e, bic_e, m_e = _fit_score(extended_parents)

    new_parents = [p for p in extended_parents if p not in baseline_parents]
    coef = getattr(m_e, 'coef_', None)
    if coef is None:
        coef = getattr(getattr(m_e, 'model_', None), 'coef_', None)
    new_coef_nonzero = bool(coef is not None and np.any(np.abs(np.asarray(coef)) > 1e-9))

    return {
        'r2_base_heldout': round(r2_b, 4), 'r2_ext_heldout': round(r2_e, 4),
        'r2_gain_heldout': round(r2_e - r2_b, 4),
        'mape_base_heldout': round(mape_b, 2), 'mape_ext_heldout': round(mape_e, 2),
        'mape_delta_heldout': round(mape_e - mape_b, 2),
        'bic_base_insample': round(bic_b, 2), 'bic_ext_insample': round(bic_e, 2),
        'bic_gain_insample': round(bic_b - bic_e, 2),
        'new_parents': new_parents, 'new_coef_nonzero': new_coef_nonzero,
        'n_train': len(train), 'n_test': len(test),
    }


def candidate_latency_backprop_edges(graph_json_path: str, available_columns=None) -> list:
    """Candidate cho loai canh L^B->L^A (latency cua callee anh huong latency
    cua caller -- CausIL, arXiv:2303.00554, Section "Inter-Service
    Dependencies") -- KHAC huong voi candidate_edges_from_graph() (CPU
    backpressure, cung chieu goi caller->callee): o day chieu SCM se la
    callee_latency-50 -> caller_latency-50, NGUOC chieu goi trong graph.

    Tra ve danh sach (caller, callee) -- nguoi goi tu lap canh SCM thanh
    (f'{callee}_latency-50', f'{caller}_latency-50') khi can, ham nay chi
    liet ke cap dich vu hop le (cung mot canh 'ai goi ai' voi
    candidate_edges_from_graph, chi khac cach dung sau do)."""
    adj, _node_types = load_graph(graph_json_path)
    edges = [(u, v) for u, vs in adj.items() for v in vs]
    if available_columns is not None:
        cols = set(available_columns)
        edges = [
            (u, v) for u, v in edges
            if f'{u}_latency-50' in cols and f'{v}_latency-50' in cols and f'{u}_workload' in cols
        ]
    return edges


def score_bic_gain(df_data: pd.DataFrame, candidate_edges: list,
                    metric: str = 'latency-50', min_rows: int = 200,
                    rho: float = 2.0) -> pd.DataFrame:
    """BIC-penalized score cho canh L^B->L^A, dung dung cong thuc CausIL
    (arXiv:2303.00554): Score = -2*log-L + rho*k*log(n). Voi nhieu Gauss,
    -2*log-L = n*log(RSS/n) + hang so (bo qua hang so vi chi so sanh
    TUONG DOI 2 model tren CUNG n, target). Khac score_r2_gain() (dung
    R2, khong phat do phuc tap): BIC UU TIEN dung o day vi day la canh
    MOI (chua co bat ky kiem chung nao truoc), can mot tieu chi chong
    overfit ngay tu buoc cham -- xem module docstring va README "Gioi han
    da biet" #7 (phat hien qua trial thang 9/2026: BIC bat duoc 59/61 node
    latency Train Ticket co he so =0, R2-gain don thuan khong the phan
    biet lien he thuc su voi nhieu do target bien do hep).

    Baseline : caller_latency-50 ~ caller_workload
    Extended : caller_latency-50 ~ caller_workload + callee_latency-50

    Tra ve DataFrame [caller, callee, bic_base, bic_ext, bic_gain, n_rows],
    sap xep GIAM DAN theo bic_gain (bic_gain = bic_base - bic_ext, duong
    nghia extended tot hon -- cung quy uoc dau voi r2_gain de dung chung
    select_edges_by_knee_point()).
    """
    rows = []
    for caller, callee in candidate_edges:
        wl_col = f'{caller}_workload'
        target_col = f'{caller}_{metric}'
        callee_col = f'{callee}_{metric}'
        needed = [wl_col, target_col, callee_col]
        if not all(c in df_data.columns for c in needed):
            continue
        sub = df_data[needed].dropna()
        if len(sub) < min_rows:
            continue
        n = len(sub)
        y = sub[target_col].values

        m1 = LinearRegression(positive=True).fit(sub[[wl_col]], y)
        rss1 = float(np.sum((y - m1.predict(sub[[wl_col]])) ** 2))
        bic_base = n * np.log(max(rss1, 1e-12) / n) + rho * 2 * np.log(n)

        m2 = LinearRegression(positive=True).fit(sub[[wl_col, callee_col]], y)
        rss2 = float(np.sum((y - m2.predict(sub[[wl_col, callee_col]])) ** 2))
        bic_ext = n * np.log(max(rss2, 1e-12) / n) + rho * 3 * np.log(n)

        rows.append({
            'caller': caller, 'callee': callee,
            'bic_base': round(bic_base, 2), 'bic_ext': round(bic_ext, 2),
            'bic_gain': round(bic_base - bic_ext, 2),
            'n_rows': n,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values('bic_gain', ascending=False).reset_index(drop=True)


def _knee_point_index(values_sorted_desc: np.ndarray) -> int:
    """Tim diem khuyu tay (knee point) tren mot day GIAM DAN: diem xa nhat so
    voi duong thang noi diem dau va diem cuoi (chuan hoa ve [0,1] x [0,1]
    truoc khi do khoang cach, de scale cua truc khong lam lech ket qua).

    Day la ky thuat elbow/knee-point pho bien -- cung tinh than voi buoc
    "knee-point" ma README/paper da dung cho G2 (g_threshold_sensitivity.py),
    ap dung lai o day cho gain cua tung canh thay vi cho nguong rui ro.

    Tra ve chi so (index) cua diem khuyu tay trong mang dau vao.
    """
    n = len(values_sorted_desc)
    if n <= 2:
        return 0
    x = np.linspace(0, 1, n)
    y = values_sorted_desc.astype(float)
    y_range = y.max() - y.min()
    y_norm = (y - y.min()) / y_range if y_range > 0 else np.zeros_like(y)

    x0, y0, x1, y1 = x[0], y_norm[0], x[-1], y_norm[-1]
    line_vec = np.array([x1 - x0, y1 - y0])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len

    dists = []
    for xi, yi in zip(x, y_norm):
        pt_vec = np.array([xi - x0, yi - y0])
        proj_len = np.dot(pt_vec, line_unit)
        proj_pt = np.array([x0, y0]) + proj_len * line_unit
        dists.append(np.linalg.norm(pt_vec - proj_pt))
    return int(np.argmax(dists))


def select_edges_by_knee_point(scores_df: pd.DataFrame, min_gain: float = 0.01,
                                gain_col: str = 'r2_gain') -> pd.DataFrame:
    """Chon canh TU DONG bang diem khuyu tay tren `gain_col` da sap xep
    giam dan, thay vi mot nguong co dinh khac nhau moi he thong.

    gain_col: ten cot dung lam tieu chi gain -- mac dinh 'r2_gain'
    (score_r2_gain(), canh CPU backpressure), truyen 'bic_gain' cho
    score_bic_gain() (canh latency backprop CausIL) -- MOT ham khuyu tay
    duy nhat cho ca 2 loai tieu chi, chi khac cot dau vao.

    min_gain: san chan duoi tuyet doi de loai canh gain am/gan 0 ngay ca
    khi chung nam "truoc" diem khuyu tay do nhieu so ngau nhien -- knee-
    point tren mot day toan gia tri nho vo nghia van co the tra ve mot chi
    so > 0. Day la RANG BUOC AN TOAN, khong phai nguong chinh.

    Tra ve subset cua scores_df (cac canh duoc chon), giu nguyen thu tu gain.
    """
    if scores_df.empty:
        return scores_df
    df = scores_df[scores_df[gain_col] > min_gain].reset_index(drop=True)
    if df.empty:
        return df
    knee_idx = _knee_point_index(df[gain_col].values)
    return df.iloc[:knee_idx + 1].reset_index(drop=True)


def validate_ood_safety(build_model_fn, df_data: pd.DataFrame, candidate_edges: list,
                         services: list, metric: str = 'cpu',
                         deltas=(5, 20, 50, 100, 150, 300),
                         injection_services=None, n_samples: int = 500,
                         min_abs_change_pct: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Quet do(<injection>_workload = base*(1+delta%)) VOI vs KHONG co
    candidate_edges trong DAG, dem sign-inversion (delta % am trong khi
    workload TANG) cho tung service/delta -- tai lap
    experiments/backpressure_edge_ood_safety_test.py o dang tham so hoa.

    build_model_fn(df_data, extra_edges: list[(str,str)]) -> dowhy.gcm model
    da fit -- nguoi goi tu quyet dinh cach dung Tier1/Tier2 cua chinh he
    thong minh (vd tu capacity_agent.CapacityAgent.train_accurate_path), ham
    nay CHI chiu trach nhiem PHAN SO SANH CO/KHONG candidate_edges va dem
    sign-inversion -- tach biet khoi cach build model cu the.

    QUAN TRONG -- do nhieu cua interventional_samples: gcm.interventional_samples
    lay mau NGAU NHIEN tu noise cua tung co che (AdditiveNoiseModel), nen ket
    qua trung binh tren n_samples mau CO SAI SO CHON MAU. Voi mot service co
    hieu ung that gan 0 (base gan 0 hoac khong bi anh huong boi canh vua them),
    sai so nay co the tu doi dau ket qua tu chay nay sang chay khac, du KHONG
    co gi thay doi ve mo hinh -- da quan sat truc tiep: cung mot loi goi,
    chay 2 lan, ra 2 tap final_edges khac nhau khi n_samples=200 khong seed.
    Hai bien phap khac phuc:
      1. `seed`: co dinh np.random.seed truoc MOI lan goi interventional_samples
         -> ket qua deterministic, chay lai ra dung 1 ket qua.
      2. `min_abs_change_pct`: chi tinh la "sign_inverted" neu |% thay doi| >=
         nguong nay (mac dinh 1.0%) -- loai nhieu quanh 0 (vd mot dich vu
         hau nhu khong bi anh huong nhung dao dong +-0.3% do sai so mau).
      3. `n_samples` mac dinh tang 200 -> 500 (khop N_PROJ cua CapacityAgent)
         de giam thanh phan phuong sai chon mau.

    Day la phan "toi uu phep do": canh chi nen o lai trong do() model neu
    KHONG lam tang sign-inversion so voi khong co canh, o BAT KY delta nao
    trong pham vi quet -- khong chi vi MAPE held-out giam (bai hoc tu chinh
    du an nay, xem docstring module).

    Tra ve DataFrame [variant, injection_service, delta_pct, service,
    metric_change_pct, sign_inverted].
    """
    from dowhy import gcm

    if injection_services is None:
        injection_services = services

    rows = []
    for with_edges, tag in [(False, 'BASELINE_no_candidate_edges'), (True, 'WITH_candidate_edges')]:
        edges_to_add = candidate_edges if with_edges else []
        model, df_sub = build_model_fn(df_data, edges_to_add)
        for inj in injection_services:
            inj_col = f'{inj}_workload'
            if inj_col not in df_sub.columns:
                continue
            base_wl = float(df_sub[inj_col].mean())
            for delta in deltas:
                target_wl = base_wl * (1 + delta / 100)
                np.random.seed(seed)
                samples = gcm.interventional_samples(
                    model, interventions={inj_col: lambda x, w=target_wl: w},
                    num_samples_to_draw=n_samples)
                for svc in services:
                    col = f'{svc}_{metric}'
                    if col not in df_sub.columns or col not in samples.columns:
                        continue
                    base_v = float(df_sub[col].mean())
                    pred_v = float(samples[col].mean())
                    chg = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0
                    rows.append({
                        'variant': tag, 'injection_service': inj, 'delta_pct': delta,
                        'service': svc, f'{metric}_change_pct': round(chg, 2),
                        'sign_inverted': chg < -abs(min_abs_change_pct),
                    })
    return pd.DataFrame(rows)


def select_scm_edges(graph_json_path: str, df_data: pd.DataFrame, services: list,
                      metric: str = 'cpu', min_gain: float = 0.01,
                      build_model_fn=None, deltas=(5, 20, 50, 100, 150, 300),
                      injection_services=None) -> dict:
    """Chay ca 4 pha thanh MOT loi goi: candidate -> R2 gain -> knee-point
    select -> (tuy chon) OOD-safety validate. Dung cho bat ky he thong nao
    co graph JSON (schema chung) + DataFrame telemetry tuong ung.

    build_model_fn: neu None, bo qua pha 4 (chi tra ve canh da qua knee-point,
    CHUA kiem chung an toan duoi can thiep -- nguoi goi phai tu chay pha 4
    truoc khi dua vao do() model that, dung mac dinh im lang bo qua buoc nay).

    Tra ve dict: {
        'candidate_edges': list, 'scores': DataFrame,
        'selected_edges': list[(caller,callee)],  # da qua pha 2+3
        'ood_safety': DataFrame hoac None,          # pha 4, neu chay
        'final_edges': list[(caller,callee)],       # sau khi loai canh
                                                      # gay sign-inversion moi
                                                      # (chi khac selected_edges
                                                      # neu ood_safety != None)
    }
    """
    candidates = candidate_edges_from_graph(graph_json_path, metric=metric,
                                             available_columns=df_data.columns)
    scores = score_r2_gain(df_data, candidates, metric=metric)
    selected_df = select_edges_by_knee_point(scores, min_gain=min_gain)
    selected_edges = [(f'{r.caller}_{metric}', f'{r.callee}_{metric}') for r in selected_df.itertuples()]

    result = {
        'candidate_edges': candidates, 'scores': scores,
        'selected_edges': selected_edges, 'ood_safety': None,
        'final_edges': selected_edges,
    }
    if build_model_fn is None:
        return result

    safety = validate_ood_safety(build_model_fn, df_data, selected_edges, services,
                                  metric=metric, deltas=deltas, injection_services=injection_services)
    result['ood_safety'] = safety

    if safety.empty:
        return result
    # Khoa so sanh PHAI gom injection_service -- neu chi (service, delta_pct),
    # mot inversion duoi injection='orders' o baseline se nguy trang che mot
    # inversion MOI duoi injection='front-end' o WITH (hai nguon can thiep
    # khac nhau, khong lien quan nhau), gay bo sot canh khong an toan.
    baseline_inv = set(
        (r.injection_service, r.service, r.delta_pct) for r in safety[
            (safety['variant'] == 'BASELINE_no_candidate_edges') & safety['sign_inverted']
        ].itertuples()
    )
    with_inv = safety[
        (safety['variant'] == 'WITH_candidate_edges') & safety['sign_inverted']
    ]
    new_inversions = with_inv[
        ~with_inv.apply(
            lambda r: (r['injection_service'], r['service'], r['delta_pct']) in baseline_inv, axis=1)
    ]
    unsafe_services = set(new_inversions['service'])
    result['final_edges'] = [
        (caller, callee) for caller, callee in selected_edges
        if callee.rsplit('_', 1)[0] not in unsafe_services
    ]
    return result


def select_latency_backprop_edges(graph_json_path: str, df_data: pd.DataFrame, services: list,
                                   min_gain: float = 0.0, build_model_fn=None,
                                   deltas=(5, 20, 50, 100, 150, 300),
                                   injection_services=None) -> dict:
    """Nhu select_scm_edges(), nhung cho loai canh L^B->L^A (CausIL, xem
    candidate_latency_backprop_edges()/score_bic_gain()) thay vi CPU
    backpressure -- 4 pha GIONG HET ve cau truc (candidate -> score ->
    knee-point -> OOD-safety), chi khac ham candidate/score va tieu chi
    gain (BIC thay R2, xem score_bic_gain() ve ly do).

    min_gain mac dinh 0.0 (khac select_scm_edges() mac dinh 0.01) vi
    bic_gain khong cung thang do voi r2_gain (BIC la hieu 2 gia tri
    n*log(RSS/n), thang do phu thuoc n va RSS, khong bi chan trong [0,1]
    nhu R2) -- san chan duoi 0 la du (bic_gain>0 da nghia la extended
    thang), nguong tuyet doi kieu 0.01 khong co y nghia tren thang BIC.

    Tra ve dict CUNG SCHEMA voi select_scm_edges(): {'candidate_edges',
    'scores' (co bic_gain thay r2_gain), 'selected_edges', 'ood_safety',
    'final_edges'} -- final_edges la list[(f'{callee}_latency-50',
    f'{caller}_latency-50')], sap dua thang vao DiGraph.add_edge().
    """
    candidates = candidate_latency_backprop_edges(graph_json_path, available_columns=df_data.columns)
    scores = score_bic_gain(df_data, candidates, metric='latency-50')
    selected_df = select_edges_by_knee_point(scores, min_gain=min_gain, gain_col='bic_gain')
    selected_edges = [(f'{r.callee}_latency-50', f'{r.caller}_latency-50') for r in selected_df.itertuples()]

    result = {
        'candidate_edges': candidates, 'scores': scores,
        'selected_edges': selected_edges, 'ood_safety': None,
        'final_edges': selected_edges,
    }
    if build_model_fn is None:
        return result

    safety = validate_ood_safety(build_model_fn, df_data, selected_edges, services,
                                  metric='latency-50', deltas=deltas, injection_services=injection_services)
    result['ood_safety'] = safety

    if safety.empty:
        return result
    baseline_inv = set(
        (r.injection_service, r.service, r.delta_pct) for r in safety[
            (safety['variant'] == 'BASELINE_no_candidate_edges') & safety['sign_inverted']
        ].itertuples()
    )
    with_inv = safety[
        (safety['variant'] == 'WITH_candidate_edges') & safety['sign_inverted']
    ]
    new_inversions = with_inv[
        ~with_inv.apply(
            lambda r: (r['injection_service'], r['service'], r['delta_pct']) in baseline_inv, axis=1)
    ]
    unsafe_services = set(new_inversions['service'])
    result['final_edges'] = [
        (callee_lat, caller_lat) for callee_lat, caller_lat in selected_edges
        if caller_lat.rsplit('_', 1)[0] not in unsafe_services
    ]
    return result


__all__ = ['candidate_edges_from_graph', 'score_r2_gain', 'select_edges_by_knee_point',
           'validate_ood_safety', 'select_scm_edges',
           'candidate_latency_backprop_edges', 'score_bic_gain', 'select_latency_backprop_edges']
