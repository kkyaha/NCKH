# -*- coding: utf-8 -*-
"""
SCM GRAPH BUILDER -- MOT ENGINE DUY NHAT CHO MOI LOAI CANH
=============================================================
Thay cho hai ham gan trung nhau truoc day (scm_edge_selector.select_scm_edges
cho CPU backpressure, select_latency_backprop_edges cho canh CausIL), module
nay khai bao MOT BANG TEMPLATE canh va chay MOT quy trinh duy nhat tren no.

Y tuong lay tu CausIL (arXiv:2303.00554): cai quyet dinh chat luong do thi
nhan qua khong phai thuat toan tim kiem, ma la RANG BUOC KHONG GIAN TIM KIEM
bang tri thuc mien ("prohibited edges") -- ablation cua ho do duoc giam SHD
>3.5 lan va nhanh hon >70 lan. O day tri thuc mien do duoc viet ra tuong
minh thanh bang EDGE_TEMPLATES thay vi nam rai rac trong code.

  scope='intra' : canh trong CUNG mot service (vd workload -> cpu).
                  Cau truc, KHONG hoc -- luon them neu co du cot du lieu.
  scope='inter' : canh giua hai service co quan he goi nhau trong do thi
                  phu thuoc.
                    direction='forward' : parent = caller, child = callee
                                          (workload lan theo chieu goi;
                                          CPU backpressure).
                    direction='reverse' : parent = callee, child = caller
                                          (latency cua callee dong gop vao
                                          latency caller -- CausIL L^B->L^A).
  learned=True  : phai qua candidate -> cham diem HELD-OUT -> knee-point ->
                  OOD-safety moi duoc vao do thi.
  learned=False : cau truc, them thang.

He thong MOI chi can khai bao bang template cua rieng no (neu schema metric
khac), khong phai viet them ham nao.
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from taxonomy_builder import load_graph  # noqa: E402
from scm_edge_selector import (  # noqa: E402
    score_edge_heldout, select_edges_by_knee_point, validate_ood_safety,
)


@dataclass(frozen=True)
class EdgeTemplate:
    """Mot loai canh duoc PHEP ton tai trong SCM (moi thu khong khai bao o
    day deu bi cam -- 'prohibited edges' theo CausIL)."""
    parent_metric: str
    child_metric: str
    scope: str = 'inter'          # 'intra' | 'inter'
    direction: Optional[str] = None  # 'forward' | 'reverse' (chi cho inter)
    learned: bool = False
    name: str = ''

    def label(self) -> str:
        return self.name or f'{self.parent_metric}->{self.child_metric}:{self.scope}/{self.direction}'


# Bang mac dinh: dung cho moi he thong theo schema '<service>_<metric>' ma
# repo nay da dung xuyen suot. Hai hang dau la Tier1/Tier2 (cau truc), hai
# hang cuoi la canh phai hoc + kiem chung.
DEFAULT_EDGE_TEMPLATES: List[EdgeTemplate] = [
    EdgeTemplate('workload', 'workload', scope='inter', direction='forward',
                 learned=False, name='tier1_workload_propagation'),
    EdgeTemplate('workload', 'cpu', scope='intra', learned=False, name='tier2_cpu'),
    EdgeTemplate('workload', 'mem', scope='intra', learned=False, name='tier2_mem'),
    EdgeTemplate('workload', 'latency-50', scope='intra', learned=False, name='tier2_latency'),
    EdgeTemplate('cpu', 'cpu', scope='inter', direction='forward',
                 learned=True, name='cpu_backpressure'),
    EdgeTemplate('latency-50', 'latency-50', scope='inter', direction='reverse',
                 learned=True, name='latency_backprop_causil'),
]


def default_mechanism_factory(metric: str) -> Callable[[], object]:
    """Tra ve factory tao estimator KHOP voi co che production se fit cho
    metric do -- de diem so o buoc chon canh phan anh dung mo hinh that,
    khong phai mot xap xi tuyen tinh khac."""
    from sklearn.linear_model import LinearRegression
    from queueing_regressor import QueueingLatencyRegressor

    if metric.startswith('latency'):
        return lambda: QueueingLatencyRegressor()
    return lambda: LinearRegression(positive=True)


def _call_pairs(graph_json_path: str) -> List[Tuple[str, str]]:
    adj, _ = load_graph(graph_json_path)
    return [(u, v) for u, vs in adj.items() for v in vs]


def structural_edges(graph_json_path: str, services: list, columns,
                     templates: List[EdgeTemplate]) -> List[Tuple[str, str]]:
    """Canh cau truc (learned=False): them thang, khong cham diem."""
    cols = set(columns)
    svc = set(services)
    out = []
    for t in templates:
        if t.learned:
            continue
        if t.scope == 'intra':
            for s in services:
                p, c = f'{s}_{t.parent_metric}', f'{s}_{t.child_metric}'
                if p in cols and c in cols:
                    out.append((p, c))
        else:
            for caller, callee in _call_pairs(graph_json_path):
                if caller not in svc or callee not in svc:
                    continue
                if t.direction == 'reverse':
                    p, c = f'{callee}_{t.parent_metric}', f'{caller}_{t.child_metric}'
                else:
                    p, c = f'{caller}_{t.parent_metric}', f'{callee}_{t.child_metric}'
                if p in cols and c in cols:
                    out.append((p, c))
    return out


def _candidates_for(template: EdgeTemplate, graph_json_path: str, services: list,
                    columns) -> List[Tuple[str, str, str, str]]:
    """Tra ve list (parent_col, child_col, child_service, parent_service) cho
    mot template learned=True."""
    cols = set(columns)
    svc = set(services)
    out = []
    for caller, callee in _call_pairs(graph_json_path):
        if caller not in svc or callee not in svc:
            continue
        if template.direction == 'reverse':
            parent_svc, child_svc = callee, caller
        else:
            parent_svc, child_svc = caller, callee
        p = f'{parent_svc}_{template.parent_metric}'
        c = f'{child_svc}_{template.child_metric}'
        wl = f'{child_svc}_workload'
        if p in cols and c in cols and wl in cols and p != c:
            out.append((p, c, child_svc, parent_svc))
    return out


def select_edges_for_template(template: EdgeTemplate, graph_json_path: str,
                              df_data: pd.DataFrame, services: list,
                              build_model_fn=None, injection_services=None,
                              min_gain: float = 0.0,
                              deltas=(5, 20, 50, 100, 150, 300)) -> dict:
    """Quy trinh 4 pha cho MOT template: candidate -> cham diem HELD-OUT ->
    knee-point -> OOD-safety.

    Tieu chi chon la `r2_gain_heldout` (cai thien R2 tren tap tai CAO chua
    thay) -- KHONG phai R2/BIC in-sample nhu truoc (xem
    scm_edge_selector.score_edge_heldout ve bang chung vi sao doi). BIC van
    duoc tinh va bao cao trong `scores` de doi chieu voi CausIL.
    """
    cands = _candidates_for(template, graph_json_path, services, df_data.columns)
    factory = default_mechanism_factory(template.child_metric)

    rows = []
    for parent_col, child_col, child_svc, parent_svc in cands:
        wl = f'{child_svc}_workload'
        sc = score_edge_heldout(
            df_data, target_col=child_col, split_col=wl,
            baseline_parents=[wl], extended_parents=[wl, parent_col],
            mechanism_factory=factory)
        if sc is None:
            continue
        sc.update({'parent_col': parent_col, 'child_col': child_col,
                   'parent_service': parent_svc, 'child_service': child_svc})
        rows.append(sc)

    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values('r2_gain_heldout', ascending=False).reset_index(drop=True)
        # Canh chi dang giu neu tham so moi THUC SU duoc dung (he so khac 0):
        # bai hoc tu README #7 -- mot co che he so 0 van co the co MAPE dep.
        scores = scores[scores['new_coef_nonzero']].reset_index(drop=True)

    selected_df = select_edges_by_knee_point(scores, min_gain=min_gain,
                                             gain_col='r2_gain_heldout') if not scores.empty else scores
    selected = [(r.parent_col, r.child_col) for r in selected_df.itertuples()] if not selected_df.empty else []

    result = {'template': template.label(), 'candidate_edges': cands, 'scores': scores,
              'selected_edges': selected, 'ood_safety': None, 'final_edges': selected}
    if build_model_fn is None or not selected:
        return result

    metric = template.child_metric
    safety = validate_ood_safety(build_model_fn, df_data, selected, services,
                                 metric=metric, deltas=deltas,
                                 injection_services=injection_services)
    result['ood_safety'] = safety
    if safety.empty:
        return result

    base_inv = set((r.injection_service, r.service, r.delta_pct) for r in safety[
        (safety['variant'] == 'BASELINE_no_candidate_edges') & safety['sign_inverted']].itertuples())
    with_inv = safety[(safety['variant'] == 'WITH_candidate_edges') & safety['sign_inverted']]
    new_inv = with_inv[~with_inv.apply(
        lambda r: (r['injection_service'], r['service'], r['delta_pct']) in base_inv, axis=1)]
    unsafe = set(new_inv['service'])
    result['final_edges'] = [(p, c) for p, c in selected
                             if c.rsplit('_', 1)[0] not in unsafe]
    return result


# ------------------------------------------------------------------
# CACHE: chi cache KET QUA CHON CANH (khong cache model da fit)
# ------------------------------------------------------------------
# Phan dat nhat cua train_accurate_path la pha OOD-safety: moi template
# learned=True phai fit lai toan bo DAG 2 lan (co/khong canh ung vien).
# Cache ket qua chon canh cat dung phan do; lan fit cuoi cung (~1-2s) van
# chay binh thuong nen model luon tuoi va khong phu thuoc pickle cua dowhy.
def _cache_key(graph_json_path: str, df_data: pd.DataFrame, services: list,
               templates: List[EdgeTemplate]) -> str:
    h = hashlib.sha256()
    try:
        with open(graph_json_path, 'rb') as f:
            h.update(f.read())
    except OSError:
        h.update(graph_json_path.encode())
    h.update(str(sorted(services)).encode())
    h.update(str([t.label() for t in templates]).encode())
    h.update(str(df_data.shape).encode())
    h.update(str(sorted(df_data.columns)).encode())
    # Tom tat noi dung du lieu (khong hash toan bo vi ton kem): tong/trung
    # binh tung cot -- du de phat hien du lieu doi.
    try:
        desc = df_data.sum(numeric_only=True).round(6).to_json()
        h.update(desc.encode())
    except Exception:
        pass
    return h.hexdigest()[:24]


def _cache_path(base_dir: str, key: str) -> str:
    return os.path.join(base_dir, f'scm_edges_{key}.json')


def build_scm_edges(graph_json_path: str, df_data: pd.DataFrame, services: list,
                    templates: Optional[List[EdgeTemplate]] = None,
                    build_model_fn=None, injection_services=None,
                    cache_dir: Optional[str] = None, use_cache: bool = True) -> dict:
    """Xay TOAN BO tap canh SCM tu bang template.

    Tra ve {'structural_edges': [...], 'learned_edges': [...],
            'all_edges': [...], 'reports': {template_label: report}}
    """
    templates = templates or DEFAULT_EDGE_TEMPLATES
    struct = structural_edges(graph_json_path, services, df_data.columns, templates)

    key = _cache_key(graph_json_path, df_data, services, templates)
    cpath = _cache_path(cache_dir, key) if cache_dir else None
    if use_cache and cpath and os.path.exists(cpath):
        try:
            with open(cpath, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            learned = [tuple(e) for e in cached['learned_edges']]
            return {'structural_edges': struct, 'learned_edges': learned,
                    'all_edges': struct + learned,
                    'reports': cached.get('reports', {}), 'from_cache': True}
        except Exception:
            pass

    learned: List[Tuple[str, str]] = []
    reports: Dict[str, dict] = {}
    for t in templates:
        if not t.learned:
            continue
        rep = select_edges_for_template(
            t, graph_json_path, df_data, services,
            build_model_fn=build_model_fn(struct + learned) if callable(build_model_fn) else None,
            injection_services=injection_services)
        learned.extend(rep['final_edges'])
        reports[t.label()] = {
            'n_candidates': len(rep['candidate_edges']),
            'n_selected': len(rep['selected_edges']),
            'n_final': len(rep['final_edges']),
            'final_edges': [list(e) for e in rep['final_edges']],
        }

    if cpath:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cpath, 'w', encoding='utf-8') as f:
                json.dump({'learned_edges': [list(e) for e in learned],
                           'reports': reports}, f, indent=2)
        except OSError:
            pass

    return {'structural_edges': struct, 'learned_edges': learned,
            'all_edges': struct + learned, 'reports': reports, 'from_cache': False}


__all__ = ['EdgeTemplate', 'DEFAULT_EDGE_TEMPLATES', 'default_mechanism_factory',
           'structural_edges', 'select_edges_for_template', 'build_scm_edges']
