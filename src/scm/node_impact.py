# -*- coding: utf-8 -*-
"""
XEP HANG NODE THEO MUC ANH HUONG NGUOI DUNG + DO ON DINH DU BAO
====================================================================
Tra loi 2 cau hoi khac nhau tren MOT Global DAG da fit
(capacity_agent.py::train_accurate_path()), ca hai deu system-agnostic
(khong gia dinh ten service/metric nao ngoai schema chung `<service>_<metric>`
va `<service>_workload` ma toan bo repo da dung):

  1. rank_user_impact(): node nao, neu bien dong, anh huong MANH NHAT den
     trai nghiem nguoi dung (mac dinh: latency tai gateway)? Do bang
     ELASTICITY -- %thay doi cua node muc tieu ung voi 1% thay doi cua
     node ung vien -- qua deterministic_forward() (KHONG Monte Carlo: cong
     thuc dong da duoc paper xac nhan tuong duong ve ky vong o RQ1
     full-resolution, nhanh va TAT DINH hon gcm.interventional_samples()
     -- quan trong vi ham nay goi forward-pass MOT LAN CHO MOI node ung
     vien, Monte Carlo se cham hang tram lan khong can thiet).

  2. evaluate_node_stability(): node nao co CO CHE DU BAO on dinh/chinh
     xac, node nao khong nen tin? Danh gia held-out THEO DUNG protocol OOD
     Gold Standard da dung cho Fast Path (train 67% gia tri THAP, test 33%
     gia tri CAO -- xem docs/HE_THONG.md (muc 6)) nhung ap dung cho
     CHINH co che cua node do trong DAG (co the nhieu cha, vd canh
     backpressure), khong phai mo hinh Bivariate rieng nhu Fast Path.

select_key_nodes() gop ca hai thanh MOT khuyen nghi: node dang du bao/theo
doi nhat la node VUA anh huong nguoi dung manh VUA co du bao on dinh. Node
anh huong manh nhung du bao KHONG on dinh duoc bao cao RIENG nhu mot canh
bao -- khong am tham loai bo, dung tinh than "khong che giau caveat" da
xuyen suot repo (vd _classify_ood_confidence, certified_envelope).
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
try:                                        # dowhy >= 0.9
    from dowhy.graph import get_ordered_predecessors
except ImportError:                         # dowhy 0.8 (ban dang cai tren may nay)
    from dowhy.gcm.graph import get_ordered_predecessors

sys.path.insert(0, os.path.dirname(__file__))
from deterministic_forward import deterministic_forward  # noqa: E402
from scm_edge_selector import _knee_point_index  # noqa: E402
from queueing_regressor import QueueingLatencyRegressor  # noqa: E402


def _mape(y_true, y_pred) -> float:
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    m = yt != 0
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100) if m.sum() > 0 else float('nan')


def _is_degenerate_mechanism(sk, atol: float = 1e-9) -> bool:
    """True neu co che DA FIT co he so hoi quy ~0 tren MOI bien dau vao --
    tuc du bao MOT HANG SO bat ke input, khong nhay voi bat ky can thiep
    nao. Phat hien tren Train Ticket (README "Gioi han da biet" #7): khi
    dieu nay xay ra tren mot target co bien do gia tri nho/hep (vd latency
    dao dong 0.01-0.05), MAPE co the VAN thap (du bao hang so gan trung
    binh, sai % tuong doi nho) du R2 am va he so =0 -- MAPE mot minh KHONG
    du de phat hien truong hop nay (dung y voi docs/HE_THONG.md (muc 6)
    ve viec khong the chi dung MAPE), nen kiem coef_ TRUC TIEP thay vi suy
    tu MAPE/R2, cung tinh than voi _check_monotone_precondition() (capacity_
    agent.py) da kiem coef_ truc tiep cho dau, o day kiem cho GAN-KHONG."""
    coef = sk.model_.coef_ if isinstance(sk, QueueingLatencyRegressor) else getattr(sk, 'coef_', None)
    if coef is None:
        return False
    return bool(np.allclose(coef, 0.0, atol=atol))


def _infer_gateway(dag_graph: nx.DiGraph) -> Optional[str]:
    """Cung mot quy tac voi taxonomy_builder.derive_primary_gateway(), ap
    dung THANG len dag_graph da fit thay vi doc lai file JSON.

    LUU Y: node cua dag_graph la node CAP DO METRIC (vd 'front-end_workload',
    'orders_cpu'), khac voi node CAP DO SERVICE cua graph JSON topology ma
    derive_primary_gateway() doc -- nen chi xet node goc dang '<service>_workload'
    (root that su cua Tier-1) roi bo suffix, thay vi tra ve ca node id."""
    roots = [n for n in dag_graph.nodes()
             if dag_graph.in_degree(n) == 0 and n.endswith('_workload')]
    if not roots:
        return None
    best = sorted(roots, key=lambda n: (-dag_graph.out_degree(n), n))[0]
    return best[:-len('_workload')]


def rank_user_impact(
    dag_graph: nx.DiGraph, model, baseline_df: pd.DataFrame,
    target_nodes: Optional[List[str]] = None, gateway: Optional[str] = None,
    delta_pct: float = 20.0,
) -> pd.DataFrame:
    """Xep hang tung cap (node, target) theo elasticity -- %thay doi cua
    `target` ung voi 1% thay doi cua `node` -- qua MOT lan forward-pass
    tat dinh cho MOI node ung vien (khong Monte Carlo; mot lan forward-pass
    cho ca gia tri cua TAT CA node cung luc, nen nhieu target khong lam
    tang so lan goi deterministic_forward()).

    target_nodes: mac dinh TOAN BO node '..._latency-50' trong do thi --
    "nguoi dung" duoc hieu la BAT KY diem latency nao request co the cham
    phai, khong chi latency tai gateway. Ly do quan trong: da kiem chung
    thuc nghiem (chay tren SockShop) rang trong Global DAG hien tai, node
    latency CHI co duy nhat CHINH workload cua chinh service do (va cac
    workload cha truoc no trong call chain) la ancestor -- KHONG CO node
    CPU/mem nao (ke ca 3 canh backpressure CPU->CPU) tung dan toi bat ky
    node latency nao, vi backpressure edges la CPU->CPU thuan tuy va dead-
    end tai CPU (khong co canh CPU->latency trong toan bo Tier-2). Neu chi
    dung MOT target la latency-50 cua gateway, ket qua suy bien ve gan nhu
    rong (front-end_latency-50 tren SockShop chi co DUY NHAT front-end_
    workload la ancestor). Dung nhieu target (latency cua moi service tren
    duong di) cho ket qua co y nghia hon, NHUNG phat hien tren van dung:
    node CPU/mem se LUON co abs_elasticity=0 (hoac bi bo qua vi khong co
    duong toi bat ky target) tren toan bo repo hien tai -- day la mot GIOI
    HAN CUA MO HINH can neu ro khi bao cao (Tier-2 chua model hoa lan
    truyen do bao hoa CPU sang do tre), khong phai loi cua ham xep hang.

    CANH BAO rieng cho Train Ticket (README "Gioi han da biet" #7): voi
    target_nodes mac dinh (moi latency-50), ham nay tra ve DataFrame RONG
    -- khong phai loi, ma vi CA 28/28 co che latency cua Train Ticket fit
    ra he so = 0 (xem queueing_regressor.py). Voi he thong nay, truyen
    target_nodes=[f'{s}_cpu' for s in services] de dung tin hieu CPU
    (co du lieu that qua backpressure edges) thay cho latency.

    Tra ve DataFrame [node, target_node, elasticity, abs_elasticity, n_hops],
    sap xep giam dan theo abs_elasticity. elasticity=inf danh dau node ma
    can thiep +delta_pct% lam target CHAM/VUOT tran dung luong rieng cua no.
    """
    if target_nodes is None:
        target_nodes = sorted(n for n in dag_graph.nodes() if n.endswith('_latency-50'))
    target_nodes = [t for t in target_nodes if t in dag_graph.nodes()]
    if not target_nodes:
        raise ValueError("Khong tim thay target_node hop le trong dag_graph -- "
                          "truyen target_nodes= tuong minh.")

    baseline_vals, _ = deterministic_forward(dag_graph, model, baseline_df, {})
    target_bases = {t: baseline_vals[t] for t in target_nodes}

    rows = []
    for node in dag_graph.nodes():
        reachable = [t for t in target_nodes if t != node and nx.has_path(dag_graph, node, t)]
        if not reachable:
            continue
        base_val = float(baseline_df[node].mean())
        if base_val == 0:
            continue  # khong the tinh %thay doi tu 0
        perturbed_val = base_val * (1.0 + delta_pct / 100.0)
        vals, breached = deterministic_forward(dag_graph, model, baseline_df, {node: perturbed_val})

        for t in reachable:
            tb = target_bases[t]
            if t in breached:
                elasticity = float('inf')
            elif tb == 0:
                continue
            else:
                elasticity = ((vals[t] - tb) / abs(tb) * 100.0) / delta_pct
            rows.append({
                'node':          node,
                'target_node':   t,
                'elasticity':    elasticity,
                'abs_elasticity': abs(elasticity),
                'n_hops':        nx.shortest_path_length(dag_graph, node, t),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values('abs_elasticity', ascending=False).reset_index(drop=True)


def summarize_node_impact(impact_df: pd.DataFrame) -> pd.DataFrame:
    """Gop impact_df (dang dai: 1 dong / cap node-target) ve 1 dong / node
    -- 'anh huong nguoi dung' cua mot node la MUC LON NHAT trong so tat ca
    target no cham toi (worst-case, khong phai trung binh: mot node chi
    anh huong 1 latency THAT MANH van dang duoc coi la 'anh huong nguoi
    dung cao', khong nen bi trung binh lam mo di boi cac target no khong
    cham toi).

    Tra ve DataFrame [node, max_abs_elasticity, worst_target, n_targets_reached].
    """
    if impact_df.empty:
        return pd.DataFrame(columns=['node', 'max_abs_elasticity', 'worst_target', 'n_targets_reached'])
    # Bo dong co abs_elasticity NaN TRUOC khi gop: idxmax() tren mot nhom
    # toan NaN tra ve NaN, sau do .loc[NaN] nem KeyError. NaN xuat hien khi
    # mot node cha bi 'breached' (gia tri inf) lan truyen xuong lam phep
    # tinh %thay doi thanh inf-inf -- gap khi latency co nhieu cha (canh
    # CausIL), truoc do khong the xay ra vi latency chi co 1 cha.
    impact_df = impact_df[impact_df['abs_elasticity'].notna()]
    if impact_df.empty:
        return pd.DataFrame(columns=['node', 'max_abs_elasticity', 'worst_target', 'n_targets_reached'])
    rows = []
    for node, sub in impact_df.groupby('node'):
        idx = sub['abs_elasticity'].idxmax()
        rows.append({
            'node':               node,
            'max_abs_elasticity': sub.loc[idx, 'abs_elasticity'],
            'worst_target':       sub.loc[idx, 'target_node'],
            'n_targets_reached':  len(sub),
        })
    return pd.DataFrame(rows).sort_values('max_abs_elasticity', ascending=False).reset_index(drop=True)


def select_top_impact_nodes(impact_df: pd.DataFrame, min_abs_elasticity: float = 0.01) -> pd.DataFrame:
    """Chon subset 'anh huong nguoi dung nhat' bang diem khuyu tay tren
    max_abs_elasticity (summarize_node_impact(), 1 dong/node) -- tai su
    dung _knee_point_index cua scm_edge_selector.py, MOT quy tac khuyu tay
    duy nhat cho ca canh SCM, services archetype, va impact node.

    Node elasticity=inf (breach tran dung luong) LUON duoc giu bat ke
    knee-point (anh huong toi da theo dinh nghia, khong can xep hang tuong
    doi voi phan con lai) -- knee-point chi ap dung cho phan huu han.

    Tra ve DataFrame cung schema voi summarize_node_impact().
    """
    if impact_df.empty:
        return impact_df
    summary = summarize_node_impact(impact_df)
    if summary.empty:
        return summary
    is_nan = summary['max_abs_elasticity'].isna()
    inf_rows = summary[np.isinf(summary['max_abs_elasticity']) & ~is_nan]
    finite_df = summary[np.isfinite(summary['max_abs_elasticity']) & ~is_nan]
    finite_df = finite_df[finite_df['max_abs_elasticity'] > min_abs_elasticity]
    if finite_df.empty:
        return inf_rows
    knee_idx = _knee_point_index(finite_df['max_abs_elasticity'].values)
    selected = finite_df.iloc[:knee_idx + 1]
    return pd.concat([inf_rows, selected]).reset_index(drop=True)


def evaluate_node_stability(
    dag_graph: nx.DiGraph, model, baseline_df: pd.DataFrame,
    split_ratio: float = 0.67, min_rows: int = 200,
) -> pd.DataFrame:
    """Danh gia do on dinh du bao cua CHINH co che DA FIT (KHONG refit mot
    ban sao moi) cho tung node co cha (root khong co co che, bo qua).

    QUAN TRONG (sua sau khi phat hien tren Train Ticket): ban dau ham nay
    REFIT mot instance model MOI tren mot split rieng, thay vi doc he so
    cua CHINH self.global_dag_model dang duoc dung de du bao thuc te. Hai
    ban fit co the LECH NHAU: train_accurate_path() fit tren mot mau con
    (vd 2000 dong random cua Train Ticket, xem capacity_agent.py) trong
    khi ham nay (ban cu) refit tren TOAN BO baseline_df voi split khac --
    da quan sat truc tiep 11/28 node latency Train Ticket "EXCELLENT/FAIR"
    theo ban refit cu, dung khi CA 28/28 co che THAT (model.causal_mechanism)
    co coef_==0 (README "Gioi han da biet" #7) -- tuc ban cu bao "dang tin"
    cho mot con so ma model THAT dang dung lai la hang so. Sua bang cach
    doc truc tiep .predict() cua co che DANG DUNG, khong .fit() lai gi ca.

    Giao thuc split GIONG Fast Path OOD Gold Standard (train_fast_path):
    sap xep theo gia tri CUA CHINH NODE do tang dan, CHI dung phan 33% CAO
    nhat lam test (67% thap con lai khong dung de fit lai -- model DA duoc
    fit tu truoc boi train_accurate_path(), o day chi DANH GIA).

    Tra ve DataFrame [node, mape_pct, r2, n_train, n_test, tag] voi tag in
    {'EXCELLENT','FAIR','POOR'} (nguong <10%/<25% GIONG train_fast_path,
    tai su dung de "on dinh" co cung y nghia voi accuracy_df hien co).
    n_train o day la kich thuoc phan 67% KHONG dung de fit lai (chi de bao
    cao ty le split), khong phai so dong THAT model da tung fit tren.
    """
    rows = []
    for node in dag_graph.nodes():
        # sorted(...) -- PHAI khop thu tu cha dowhy dung khi fit, xem
        # deterministic_forward.py (cung bug, cung fix).
        parents = get_ordered_predecessors(dag_graph, node)
        if not parents:
            continue
        cols = parents + [node]
        if not all(c in baseline_df.columns for c in cols):
            continue
        sub = baseline_df[cols].dropna()
        if len(sub) < min_rows:
            continue

        sub_sorted = sub.sort_values(node).reset_index(drop=True)
        split = int(len(sub_sorted) * split_ratio)
        train, test = sub_sorted.iloc[:split], sub_sorted.iloc[split:]
        if len(test) < 10:
            continue

        sk = model.causal_mechanism(node).prediction_model.sklearn_model
        y_pred = sk.predict(test[parents].values)
        y_true = test[node].values

        mape_v = _mape(y_true, y_pred)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2_v = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        # coef_~=0 -> POOR bat ke MAPE noi gi: mot co che du bao HANG SO co
        # the van co MAPE thap tren target bien do hep (xem
        # _is_degenerate_mechanism) -- day la dieu MAPE-only bo sot tren
        # Train Ticket, phai kiem TRUOC roi moi xet nguong MAPE binh thuong.
        if _is_degenerate_mechanism(sk):
            tag = 'POOR'
        else:
            tag = 'EXCELLENT' if mape_v < 10 else ('FAIR' if mape_v < 25 else 'POOR')

        rows.append({
            'node': node, 'mape_pct': round(mape_v, 2), 'r2': round(r2_v, 3),
            'n_train': len(train), 'n_test': len(test), 'tag': tag,
        })
    return pd.DataFrame(rows)


def select_key_nodes(
    impact_df: pd.DataFrame, stability_df: pd.DataFrame,
    root_nodes: Optional[set] = None,
    min_abs_elasticity: float = 0.01, unstable_tag: str = 'POOR',
) -> Dict[str, list]:
    """Gop rank_user_impact() + evaluate_node_stability() thanh MOT khuyen
    nghi node nao nen dung de du bao/theo doi:

      - high_impact            : select_top_impact_nodes() -- anh huong
                                  nguoi dung manh, CHUA xet do on dinh.
      - recommended            : high_impact MA CO danh gia on dinh va tag
                                  != unstable_tag (mac dinh != 'POOR'), HOAC
                                  la node goc (root_nodes -- vd workload
                                  cua gateway, doc truc tiep tu telemetry,
                                  KHONG qua mot co che du bao nao nen
                                  khong co 'do on dinh du bao' de danh gia
                                  -- tin cay theo chat luong DU LIEU, khong
                                  phai chat luong MO HINH).
      - unstable_but_impactful : high_impact, KHONG phai root, VA
                                  (tag==unstable_tag HOAC khong danh gia
                                  duoc do thieu du lieu held-out) -- canh
                                  bao RIENG, khong am tham loai bo: anh
                                  huong that nhung con so du bao cho no
                                  khong dang tin, phai neu ro khi bao cao.
    """
    top_impact = select_top_impact_nodes(impact_df, min_abs_elasticity)
    if top_impact.empty:
        return {'high_impact': [], 'recommended': [], 'unstable_but_impactful': []}

    stability_by_node = {r.node: r.tag for r in stability_df.itertuples()} if not stability_df.empty else {}
    root_nodes = root_nodes or set()

    high_impact = top_impact['node'].tolist()
    recommended, unstable = [], []
    for node in high_impact:
        if node in root_nodes:
            recommended.append(node)
            continue
        tag = stability_by_node.get(node)
        if tag is None or tag == unstable_tag:
            unstable.append(node)
        else:
            recommended.append(node)

    return {
        'high_impact': high_impact,
        'recommended': recommended,
        'unstable_but_impactful': unstable,
    }


__all__ = [
    'rank_user_impact', 'summarize_node_impact', 'select_top_impact_nodes',
    'evaluate_node_stability', 'select_key_nodes',
]
