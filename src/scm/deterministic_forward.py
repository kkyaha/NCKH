# -*- coding: utf-8 -*-
"""
DETERMINISTIC FORWARD PASS TREN GLOBAL DAG (system-agnostic)
================================================================
Mot lan duy nhat cho phep tinh he so tat dinh, khong-Monte-Carlo cua
Global Causal DAG: di qua do thi theo thu tu topo, goi .predict() cua
tung co che sklearn da fit voi gia tri diem cua cha no (khong lay mau
noise cong them). Duoc dung boi:

  - capacity_agent.py: certified_envelope()/joint_certified_envelope()
    (Proposition 2/3) -- can DUNG mot lan chay tat dinh, khong duoc dung
    gcm.interventional_samples() (lay mau ngau nhien, pha vo tinh
    certificate -- xem docstring capacity_agent.py).
  - node_impact.py: rank_user_impact() -- can hang tram lan forward-pass
    (moi node ung vien 1 lan) de do elasticity; Monte Carlo (500 mau/lan)
    se dat gap ~100x cham hon khong can thiet cho mot con so trung binh
    tat dinh da co cong thuc dong (dung cong thuc paper da xac nhan tuong
    duong o RQ1: E[Target|do(w)] = model.predict(w), xem
    docs/RQ1_SCM_ACCURACY_REPORT.md).

Truoc day logic nay lap lai 2 lan gan giong het nhau trong capacity_agent.py
(`_deterministic_forward` cho 1 injection, `_deterministic_forward_multi`
cho nhieu injection dong thoi) -- gop ve MOT ham nhan dict injections (co
the rong hoac chi 1 phan tu) de tranh code trung.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from dowhy.graph import get_ordered_predecessors

sys.path.insert(0, os.path.dirname(__file__))
from queueing_regressor import QueueingLatencyRegressor  # noqa: E402


def deterministic_forward(
    dag_graph: nx.DiGraph,
    model,
    baseline_df,
    injections: Dict[str, float],
) -> Tuple[Dict[str, float], List[str]]:
    """Lan truyen tat dinh qua DAG theo thu tu topo.

    injections: {node: value} -- node nao co trong day nhan dung gia tri
    da cho (ghi de len co che rieng cua no); node khong co trong day duoc
    tinh tu cha no qua .predict() (root khong cha -> trung binh baseline).

    Tra ve (values, breached): `breached` liet ke cac node latency ma MOT
    TRONG SO CAC cha (khong chi cha dau tien) da cham/vuot tran dung luong
    RIENG CUA CHINH CANH DO (bien mien xac dinh cua phep bien doi hang doi
    phi(x)=x/(c-x), QueueingLatencyRegressor tinh capacity_ rieng cho MOI
    cot dau vao) -- duoc bao ro thay vi ngoai suy am tham qua mot ham hang
    doi phan ky.

    SUA LOI (phat hien khi them canh CausIL L^B->L^A khien latency node co
    2 cha thay vi 1): ban truoc CHI kiem tra breach khi `X.shape[1]==1`,
    ngam dinh moi latency mechanism chi co dung 1 cha (workload). Voi 2+
    cha, dieu kien do luon False -- kiem tra tran dung luong bi BO QUA AM
    THAM, lam Proposition 2 (certified_envelope) mat kha nang gan co
    CEILING_BREACHED cho dung nhung node vua duoc them canh. Sua bang cach
    kiem TUNG cot cua X rieng le, khong con phu thuoc so luong cha.

    Dung chung cho ca (a) mot injection (certified_envelope) va (b) nhieu
    injection dong thoi (joint_certified_envelope) -- truyen {} de lay
    hoan toan gia tri baseline (dung lam diem chuan cho impact ranking).
    """
    values: Dict[str, float] = {}
    breached: List[str] = []
    for node in nx.topological_sort(dag_graph):
        if node in injections:
            values[node] = injections[node]
            continue
        # PHAI dung dung thu tu cha ma dowhy DA DUNG KHI FIT
        # (dowhy.graph.get_ordered_predecessors = sorted(predecessors)).
        # Truoc day dung list(predecessors) = thu tu CHEN canh vao DiGraph,
        # khac thu tu sorted bat cu khi nao node co >=2 cha -- X bi hoan vi
        # cot so voi luc fit, .predict() tra ve rac. Bug nay am tham tu lau
        # (moi node CPU co canh backpressure deu dinh) nhung chi lo ra khi
        # them canh latency nhieu cha, noi hai cot lech nhau ~35 lan do lon
        # (workload ~0.6 vs latency ~0.017) khien MAPE vot len 27000%.
        parents = get_ordered_predecessors(dag_graph, node)
        if not parents:
            values[node] = float(baseline_df[node].mean())
            continue
        mech = model.causal_mechanism(node)
        sk = mech.prediction_model.sklearn_model
        X = np.array([[values[p] for p in parents]])
        if isinstance(sk, QueueingLatencyRegressor):
            # Kiem TUNG cot rieng (khong chi cot dau) -- capacity_ la mang
            # cung so chieu voi X, moi cot co tran dung luong rieng cua no.
            if any(float(X[0, i]) >= float(sk.capacity_[i]) for i in range(X.shape[1])):
                breached.append(node)
                values[node] = float('inf')
                continue
        values[node] = float(sk.predict(X)[0])
    return values, breached


__all__ = ['deterministic_forward']
