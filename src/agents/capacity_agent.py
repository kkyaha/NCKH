# -*- coding: utf-8 -*-
"""
CapacityAgent — SCM Capacity & Breakpoint Reasoner (Hệ A, cơ chế DUY NHẤT)
===========================================================================
Một Agent duy nhất tuân thủ mô hình ReAct (Yao et al., 2022), dùng MỘT cơ chế
tính toán (Global Causal DAG) cho mọi đầu ra -- không còn mô hình bivariate
rời chạy song song (đã bỏ, xem lịch sử git):

  1. ACT (Hành động qua Tool):
     - Siêu đồ thị 28-node Causal DAG (mô phỏng lan truyền suy giảm cascade)
       + "correction theo tầng" (chain-aware scaling, _apply_chain_correction).

  2. OBSERVE (Quan sát định lượng):
     - Tổng hợp độ lệch tài nguyên (CPU, Memory, Latency, Socket).
     - Quét biên dự phòng so với trần P99 huấn luyện (compute_headroom/summarize_headroom).
     - find_breakpoint(): quét tải tìm điểm gãy R* (SAFE->CRITICAL), không cần
       biết trước tải đỉnh kỳ vọng (L_peak) -- tự báo cáo R* của chính nó.

  3. REASON & CRITIQUE (Suy luận chuyên gia & Tự phản biện Devil's Advocate):
     - Phân tích nguyên nhân điểm nghẽn (Bottleneck Attribution).
     - Phân tích hiệu ứng suy giảm theo bước nhảy (Cascade Attenuation).
     - Tự phản biện "Devil's Advocate" tìm các rủi ro tiềm ẩn bị bỏ sót (OOD, connection pool, v.v.).
     - Đưa ra khuyến nghị kỹ thuật hành động được (Actionable Mitigations: Caching, Scaling, Circuit Breakers).
"""

import os
import sys
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

# Resolve paths
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_AGENT_DIR)
BASE_DIR   = os.path.dirname(_SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))

from data_processor import (
    load_multi_service_data,
    split_quantile,
    SERVICES,
    TRAINTICKET_SERVICES,
    ONLINEBOUTIQUE_SERVICES,
)
from queueing_regressor import QueueingLatencyRegressor
from deterministic_forward import deterministic_forward
from taxonomy_builder import load_graph, derive_primary_gateway
from scm_graph_builder import build_scm_edges, DEFAULT_EDGE_TEMPLATES
from node_impact import (
    rank_user_impact as _rank_user_impact,
    evaluate_node_stability as _evaluate_node_stability,
    select_key_nodes as _select_key_nodes,
)


# ============================================================
# 1. NON-LINEAR QUEUEING REGRESSOR (Dành cho Latency)
# ============================================================
# Chuyen sang src/scm/queueing_regressor.py (import o dau file) de
# deterministic_forward.py dung chung duoc ma khong pha phan tang
# src/scm/ -> src/agents/ (xem README "Vi sao tach vay"). Ten giu nguyen
# qua import, hanh vi khong doi.


# ============================================================
# 2. OUTPUT DATA STRUCTURE: CapacityAssessment
# ============================================================
@dataclass
class CapacityAssessment:
    """
    Kết quả đánh giá năng lực toàn diện của CapacityAgent (ReAct Pattern).
    """
    status: str                         # "SAFE" | "WARNING" | "CRITICAL"
    simulation_result: Dict[str, Any]   # Global DAG multi-hop cascade (cơ chế DUY NHẤT)
    saturated_services: List[str]       # Danh sách service vượt ngưỡng an toàn (>70%)
    expert_assessment: str              # Phân tích nguyên nhân điểm nghẽn từ LLM / Rule
    risk_critique: str                  # Tự phản biện Devil's Advocate (rủi ro tiềm ẩn)
    recommendations: List[str]          # Các khuyến nghị kỹ thuật cụ thể
    confidence: str = "MEDIUM"          # "HIGH" | "MEDIUM" | "LOW"
    ood_confidence: str = "high"        # G7: "high"|"medium"|"low"|"very_low" — mức tin cậy
                                         # ngoại suy của *độ lớn* workload dự phóng, độc lập
                                         # với `confidence` (vốn chỉ đo taxonomy-membership).
    ood_flagged_nodes: List[str] = field(default_factory=list)  # node nào rơi ngoài P95 train
    headroom: Dict[str, Any] = field(default_factory=dict)      # xem compute_headroom()


# ============================================================
# 3. CAPACITY AGENT (UNIFIED PERFORMANCE & SIMULATION AGENT)
# ============================================================
class CapacityAgent:
    """
    Agent Đánh Giá Năng Lực & Hiệu Năng Hệ Thống (ReAct Agent).

    Tích hợp:
      - Siêu đồ thị 28-node Causal DAG phản ánh cascade attenuation (cơ chế tính toán DUY NHẤT).
      - find_breakpoint(): quét nhị phân trên DAG trên để tìm điểm gãy R*.
      - ReAct Loop: Act (chạy SCM) -> Observe (tổng hợp delta) -> Reason & Critique (LLM/Expert logic).
    """

    def __init__(
        self,
        llm: Any = None,
        data_dir: str = None,
        graph_path: str = None,
        system_type: str = 'sockshop',
        services: List[str] = None,
        auto_train: bool = True
    ):
        self.system_type = system_type
        if system_type == 'trainticket' or (data_dir and 'trainticket' in data_dir.lower()):
            self.system_type = 'trainticket'
            if data_dir is None:
                data_dir = os.path.join(BASE_DIR, 'data', 'raw', 'trainticket')
            if graph_path is None:
                graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'trainticket_agent_graph.json')
            self.services = services or TRAINTICKET_SERVICES
        elif system_type == 'onlineboutique' or (data_dir and 're2-ob' in data_dir.lower()):
            # He thu BA (RCAEval RE2-OB). Khong can sua gi trong duong NAP du
            # lieu: load_multi_service_data() nhan dinh dang theo FILE THUC TE
            # (_detect_metric_file_format) va load_trainticket_data() tim thu muc
            # scenario theo su ton tai cua metrics.parquet + inject_time.txt chu
            # khong theo tien to ten -- ca hai da duoc viet san cho truong hop nay.
            #
            # Vi sao them he nay: hai phat hien ve QUY MO (ti phan nhu cau giam
            # 81% -> 15,7%; don vi quy gan node -> canh, 1x -> 32x) hien chi co
            # HAI diem tren truc quy mo (7 va 28 service). OB co 11 service
            # trong metrics -> diem GIUA. Xem papers/p2_khop_tang/docs/PHAT_HIEN_KHOP_TANG.md muc 6
            # va 2b.
            self.system_type = 'onlineboutique'
            if data_dir is None:
                data_dir = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-OB')
            if graph_path is None:
                graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'onlineboutique_agent_graph.json')
            self.services = services or ONLINEBOUTIQUE_SERVICES
        else:
            if data_dir is None:
                data_dir = os.path.join(BASE_DIR, 'data', 'raw')
            if graph_path is None:
                graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
            self.services = services or SERVICES

        self.llm        = llm
        self.data_dir   = data_dir
        self.graph_path = graph_path

        # default_injection: SUY tu graph (derive_primary_gateway), khong
        # go tay nua -- truoc day la 'front-end'/'ts-preserve-service' co
        # dinh theo system_type. 'front-end' (SockShop) trung voi gateway
        # that (khong doi hanh vi). 'ts-preserve-service' (Train Ticket) LA
        # mot seed archetype (BOOK_TICKET), KHONG PHAI gateway that (gateway
        # that la 'ts-ui-dashboard') -- doi sang gateway that dung voi paper
        # (Section "Scope": "applied at a gateway"). self.gateways giu toan
        # bo gateway hop le (moi node type=='gateway') cho nhu cau multi-
        # gateway sau nay, self.default_injection la gateway CHINH (tuong
        # thich nguoc voi moi cho dang dung 1 string).
        try:
            _adj, _node_types = load_graph(self.graph_path)
            self.gateways = sorted(n for n, t in _node_types.items() if t == 'gateway')
            self.default_injection = derive_primary_gateway(_adj, _node_types)
        except (FileNotFoundError, ValueError):
            self.gateways = []
            self.default_injection = self.services[0] if self.services else None

        # State của Tool Accurate Path (Global DAG, cơ chế DUY NHẤT)
        self.global_dag_model = None
        self.global_model     = None
        self.global_df_baseline: pd.DataFrame = None
        self.df_baseline      = None
        self.dag_graph: nx.DiGraph = None

        # G7 state (see simulate_intervention / _classify_ood_confidence)
        self._ood_stats_cache = None
        self._last_ood_confidence = "high"
        self._last_ood_flagged: List[str] = []
        self.dag              = None

        # Node confidence: select_key_nodes() cache (tu dong, xem
        # _get_key_nodes) + co che danh dau TAY mot node la khong dang tin
        # (vd biet truoc tu G7/README, khong can cho elasticity/stability
        # tu dong bat duoc) -- xem mark_node_unreliable().
        self._key_nodes_cache: Dict[str, List[str]] = None
        self._stability_cache: pd.DataFrame = None
        self._manual_unreliable_nodes: Dict[str, str] = {}

        # Correction chain-awareness cho assess_capacity() -- xem
        # _apply_chain_correction(). Thieu file -> tat correction, HANH VI Y
        # HET truoc day (khong bao gio raise). CHI dung trong assess_capacity(),
        # KHONG dung trong simulate_intervention() -- de khong lam lech so
        # RQ1-RQ4 da cong bo (cac script do goi simulate_intervention() truc
        # tiep). Xem papers/p1_du_phong/docs/DATA_FRAMEWORK.md muc 2/5m(d).
        self._chain_correction = None
        _cc_path = os.path.join(BASE_DIR, 'data', 'processed', 'frozen', 'chain_correction_params.json')
        if os.path.isfile(_cc_path):
            try:
                with open(_cc_path, encoding='utf-8') as f:
                    self._chain_correction = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._chain_correction = None

        self._is_trained = False

        print(f"[CapacityAgent] Khởi tạo ({self.system_type}). Data dir: {self.data_dir}")
        if auto_train:
            self.train()

    # ----------------------------------------------------------
    # HUẤN LUYỆN SCM TOOLS (CẢ 2 CHẾ ĐỘ)
    # ----------------------------------------------------------
    def train(self):
        """Huấn luyện Siêu Đồ Thị Nhân Quả (Global 2-Tier Causal DAG) -- cơ chế
        tính toán DUY NHẤT của Hệ A, không còn mô hình bivariate rời chạy song song."""
        self.train_accurate_path()
        self._is_trained = bool(self.global_dag_model)
        self._key_nodes_cache = None  # DAG moi -> bo cache select_key_nodes() cu
        self._stability_cache = None  # DAG moi -> bo cache evaluate_node_stability() cu

    def train_accurate_path(self):
        """Huấn luyện Siêu Đồ Thị Nhân Quả (Global 2-Tier Causal DAG)."""
        print("\n" + "=" * 70)
        print(f"  [CapacityAgent: Tool 2] HUẤN LUYỆN GLOBAL CAUSAL DAG ({self.system_type.upper()})")
        print("=" * 70)

        df_data = load_multi_service_data(self.data_dir, system_type=self.system_type, services=self.services)
        if df_data is None or df_data.empty:
            print("[WARNING] Không load được dữ liệu đa dịch vụ.")
            return

        if not os.path.exists(self.graph_path):
            print(f"[WARNING] Không tìm thấy file graph: {self.graph_path}")
            return

        # Toan bo tap canh SCM duoc dung bang MOT engine duy nhat tren bang
        # template (src/scm/scm_graph_builder.py) -- khong con doan code rieng
        # cho tung tier/tung he thong:
        #   - Tier 1 (workload -> workload theo chieu goi)      : cau truc
        #   - Tier 2 (workload -> cpu/mem/latency cung service) : cau truc
        #   - CPU backpressure (cpu -> cpu theo chieu goi)      : phai hoc
        #   - Latency backprop (latency callee -> latency caller,
        #     CausIL arXiv:2303.00554)                          : phai hoc
        # Hai loai "phai hoc" deu qua cung mot quy trinh: candidate tu do thi
        # phu thuoc -> cham diem HELD-OUT (67/33 tai thap -> tai cao) ->
        # knee-point -> OOD-safety. Xem scm_edge_selector.score_edge_heldout()
        # ve ly do bo tieu chi in-sample (R2-gain/BIC) da dung truoc day.
        def _model_builder_for(edges_so_far):
            g_base = nx.DiGraph()
            g_base.add_edges_from(edges_so_far)
            return self._build_extended_dag_fn(g_base, df_data)

        edge_build = build_scm_edges(
            self.graph_path, df_data, self.services,
            templates=DEFAULT_EDGE_TEMPLATES,
            build_model_fn=_model_builder_for,
            injection_services=self.gateways or [self.default_injection],
            cache_dir=os.path.join(BASE_DIR, 'data', 'processed', 'scm_cache'),
        )
        self._last_edge_build = edge_build

        g = nx.DiGraph()
        # CHI DUNG structural_edges (Tier-1 + Tier-2), KHONG dung all_edges.
        # learned_edges (Tier-2.5 `R->R` backpressure) da bi LOAI BO khoi DAG
        # san pham tu 2026-10-10 -- xem comment dai o train_accurate_path()
        # (duoi day) va papers/p1_du_phong/docs/HE_THONG.md muc 4 ve ly do:
        # tren du lieu tai THAT (SS-TRAIN), quy trinh chon canh tu dong chon
        # 0/8 ung vien (R2 hai tang da 0,74-0,9972, khong con gi de T2.5
        # "giai thich them"); tren RCAEval (R2 hai tang ~0,015) no tung duoc
        # chon vi lap day khoang trong bang tuong quan gia, roi bi bac boi
        # can thiep that (132x te hon). build_scm_edges() VAN tinh
        # learned_edges (dung cache, khong ton thoi gian) de script khac
        # (vd cac rq6_*.py cua bai 2, experiments/edges/select_scm_edges.py)
        # con doc duoc edge_build['learned_edges'] nhu truoc -- chi DAG cua
        # CHINH CapacityAgent nay khong con cong no vao nua.
        g.add_edges_from(edge_build['structural_edges'])
        if edge_build.get('from_cache'):
            print(f"  [Edges] dung cache (chi lay structural, bo "
                  f"{len(edge_build['learned_edges'])} canh hoc duoc).")
        for label, rep in edge_build['reports'].items():
            print(f"  [{label}] {rep['n_candidates']} candidate -> {rep['n_selected']} qua "
                  f"held-out/knee-point -> {rep['n_final']} qua OOD-safety.")

        valid_nodes = [n for n in g.nodes() if n in df_data.columns]

        # CHOT AN TOAN (system-agnostic): node cua do thi KHONG co cot metric
        # tuong ung se bi loai am tham o dong tren. Neu node bi loai la
        # GATEWAY thi toan bo ket qua vo nghia -- moi can thiep deu ap tai
        # gateway (Gateway Invariance, Prop. 1) nen mat no la mat diem vao.
        # Phat hien khi trien khai thu sang Online Boutique: bang metrics goi
        # gateway la 'frontend' con do thi (trich tu trace, cot serviceName)
        # goi la 'frontendservice' -- hai nguon telemetry CUA CUNG HE THONG
        # dung hai quy uoc ten khac nhau. Truoc khi co chot nay, he van chay
        # het va bao "[OK] Da khop Global DAG 24 nodes" trong khi DAG thieu
        # han gateway; day dung loai loi phai bao to, khong phai bo qua.
        missing_gw = [gw for gw in (self.gateways or [self.default_injection])
                      if gw and f"{gw}_workload" not in df_data.columns]
        if missing_gw:
            print(f"[CANH BAO] Gateway {missing_gw} khong co cot '<gateway>_workload' trong "
                  f"telemetry -- kiem tra ten service giua do thi va bang metrics co khop "
                  f"khong. simulate_intervention() tai cac gateway nay se tra ve rong.")
        dropped = [n for n in g.nodes() if n not in df_data.columns]
        if dropped:
            print(f"[CANH BAO] {len(dropped)} node cua do thi bi loai vi thieu cot metric: "
                  f"{sorted(dropped)[:8]}{' ...' if len(dropped) > 8 else ''}")

        g_sub = g.subgraph(valid_nodes).copy()
        df_sub = df_data[valid_nodes].dropna()

        # Baseline dữ liệu bình thường
        self.global_df_baseline = df_sub
        self.df_baseline      = self.global_df_baseline
        self.dag_graph        = g_sub
        self.dag              = self.dag_graph

        # rho_s = ti le workload cua service s so voi gateway (hoi quy qua
        # goc tren baseline) -- dung lam dau vao cho _apply_chain_correction(),
        # xem papers/p1_du_phong/experiments/model_eval/evaluate_chain_awareness.py
        # (evaluate_structural_correction/fit_final_correction) ve nguon goc
        # cong thuc va so lieu kiem chung.
        gw_col = f'{self.default_injection}_workload' if self.default_injection else None
        self.rho: Dict[str, float] = {}
        if gw_col and gw_col in self.df_baseline.columns:
            gw = self.df_baseline[gw_col]
            denom = float((gw * gw).sum())
            if denom > 0:
                for s in self.services:
                    col = f'{s}_workload'
                    if col in self.df_baseline.columns:
                        self.rho[s] = float((self.df_baseline[col] * gw).sum() / denom)

        # Đảm bảo g_sub là một Causal DAG nghiêm ngặt (không có chu trình)
        while not nx.is_directed_acyclic_graph(g_sub):
            cycle = nx.find_cycle(g_sub, orientation='original')
            g_sub.remove_edge(cycle[-1][0], cycle[-1][1])

        df_fit = df_sub.sample(min(2000, len(df_sub)), random_state=42) if len(df_sub) > 2000 else df_sub

        model = gcm.InvertibleStructuralCausalModel(g_sub)
        gcm.auto.assign_causal_mechanisms(model, df_fit)
        self._assign_dag_mechanisms(model, g_sub)

        gcm.fit(model, df_fit)
        self.global_dag_model = model
        self.global_model     = self.global_dag_model
        print(f"[OK] Đã khớp Global DAG {len(valid_nodes)} nodes ({len(g_sub.edges())} edges).")

    def _assign_dag_mechanisms(self, model, g_sub: nx.DiGraph):
        """Gán cơ chế chuyên biệt: CPU/Mem + Tier-1 Workload->Workload
        (Linear, ràng buộc hệ số KHÔNG ÂM), Latency (Queueing phi tuyến).
        Tách thành method riêng (trước đây viết thẳng trong train_accurate_
        path) để dùng LẠI đúng một chỗ cho cả DAG sản phẩm và cho
        build_model_fn của scm_edge_selector.validate_ood_safety (pha 4) --
        hai nơi PHẢI gán cùng quy tắc, không được lệch nhau.

        ĐỒNG BỘ VỚI evaluation_suite.build_and_train_global_dag() (Sock Shop):
        ban đầu ở đây chỉ có _cpu/_mem dùng LinearRegression() THƯỜNG (không
        positive=True), và cạnh Tier-1 (Workload->Workload) không được gán
        ràng buộc gì cả — tức là CHÍNH XÁC bug sign-inversion mà Section
        "Extrapolation-Sign Failure Mode" của paper mô tả đã "fix" (nhưng
        fix đó trước đây chỉ nằm ở evaluation_suite.py, một script riêng
        cho RQ4, KHÔNG nằm trong CapacityAgent — class này mới là code thật
        được orchestrator.py dùng cho cả Sock Shop LẪN Train Ticket). Áp
        dụng đúng một ràng buộc cho cả 2 hệ thống ở đây để kết quả giữa
        Sock Shop và Train Ticket (vd RQ6 Part A.3) không còn lệch nhau vì
        một confound về quy trình fit, chỉ còn lệch vì bản chất dữ liệu.
        """
        # Tier-2.5 (`CoCheCongDai`, canh R->R backpressure) DA BI LOAI khoi
        # DAG san pham tu 2026-10-10 (xem comment dai o g.add_edges_from phia
        # tren) -- nen moi node _cpu/_mem o day CHI con cha workload (Tier-2
        # thuan), khong con node nao co "ca cha workload LAN cha tai nguyen"
        # nua. Nhanh CoCheCongDai (tung dung chi_so_cha_workload de tim cot
        # cha tai nguyen can kep) khong con duong nao toi duoc -- bo han,
        # khong giu lai duoi dang code chet. Lich su day du + so lieu (132x
        # te hon duoi can thiep, sau kep van gap 2,4x hai tang o mean) nam o
        # papers/p1_du_phong/docs/HE_THONG.md muc 4 va src/scm/co_che_cong_dai.py
        # (file do van con, chi khong con duoc goi tu day).
        for node in g_sub.nodes():
            if node.endswith('_cpu') or node.endswith('_mem'):
                model.set_causal_mechanism(
                    node,
                    AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True)))
                )
            elif node.endswith('_latency-50'):
                model.set_causal_mechanism(
                    node,
                    AdditiveNoiseModel(SklearnRegressionModel(QueueingLatencyRegressor()))
                )
            elif node.endswith('_workload') and g_sub.in_degree(node) > 0:
                model.set_causal_mechanism(
                    node,
                    AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True)))
                )

    def _build_extended_dag_fn(self, base_graph: nx.DiGraph, df_data: pd.DataFrame):
        """Trả về build_model_fn(df, extra_edges) -> (model, df_sub) dùng
        cho scm_edge_selector.validate_ood_safety() (pha 4): thêm extra_edges
        vào BẢN SAO của base_graph (Tier1+Tier2, chưa có backpressure), phá
        cycle nếu có, gán mechanism ĐÚNG quy tắc _assign_dag_mechanisms(),
        fit trên cùng df_data. Tách rời base_graph (không đụng vào `g` đang
        xây dở trong train_accurate_path) để mỗi lần gọi build_model_fn là
        một model độc lập, đúng hợp đồng validate_ood_safety cần."""
        def build_model_fn(df_for_build: pd.DataFrame, extra_edges: list):
            g2 = base_graph.copy()
            for u, v in extra_edges:
                g2.add_edge(u, v)
            while not nx.is_directed_acyclic_graph(g2):
                cycle = nx.find_cycle(g2, orientation='original')
                g2.remove_edge(cycle[-1][0], cycle[-1][1])

            valid = [n for n in g2.nodes() if n in df_for_build.columns]
            g2_sub = g2.subgraph(valid).copy()
            df_sub2 = df_for_build[valid].dropna()
            df_fit2 = (df_sub2.sample(min(2000, len(df_sub2)), random_state=42)
                       if len(df_sub2) > 2000 else df_sub2)

            m2 = gcm.InvertibleStructuralCausalModel(g2_sub)
            gcm.auto.assign_causal_mechanisms(m2, df_fit2)
            self._assign_dag_mechanisms(m2, g2_sub)
            gcm.fit(m2, df_fit2)
            return m2, df_sub2
        return build_model_fn

    def _compute_hops(self, injection_service: str) -> Dict[str, int]:
        """Đếm số hop từ injection_service đến mỗi service trong DAG."""
        inj_col = f"{injection_service}_workload"
        hops    = {}
        dag     = self.dag_graph if self.dag_graph is not None else self.dag
        if dag is None:
            return {s: -1 for s in self.services}
        for svc in self.services:
            tgt_col = f"{svc}_workload"
            if inj_col == tgt_col:
                hops[svc] = 0
            elif dag.has_node(inj_col) and dag.has_node(tgt_col) and nx.has_path(dag, inj_col, tgt_col):
                hops[svc] = nx.shortest_path_length(dag, inj_col, tgt_col)
            else:
                hops[svc] = -1
        return hops

    # ----------------------------------------------------------
    # G7: OOD-CONFIDENCE GUARD (magnitude-of-extrapolation check)
    # ----------------------------------------------------------
    # G6 (parser_agent.py) refuses when a requirement matches NO calibrated
    # archetype at all. G2 (parser_agent.py) separately clamps the injection
    # delta itself to [5%, 50%]. Neither protects against a downstream node
    # -- reached via Tier-1 propagation or Tier-2 conversion, not the
    # injection point itself -- landing outside ITS OWN training envelope
    # even when the injection delta is fully within G2's legal range (see
    # experiments/g7_ood_guard_test.py and Section "Extrapolation-Sign
    # Failure Mode" in the paper). G7 measures that gap directly: it is a
    # read-only classification over already-computed `samples`, not a
    # re-simulation, so it adds negligible latency to the accurate path.
    _OOD_CONF_ORDER = {'high': 0, 'medium': 1, 'low': 2, 'very_low': 3}

    def _ood_train_stats(self) -> Dict[str, Dict[str, float]]:
        """Per-node training-distribution percentiles, cached on first use."""
        if getattr(self, '_ood_stats_cache', None) is not None:
            return self._ood_stats_cache
        stats = {}
        if self.global_df_baseline is not None:
            for col in self.global_df_baseline.columns:
                vals = self.global_df_baseline[col].dropna().values
                if len(vals) > 0:
                    stats[col] = {
                        'max': float(np.max(vals)),
                        # p50 dung lam MOC DUOI cua bien du phong (xem
                        # compute_headroom): moc duoi va tran phai cung ho
                        # thong ke -- dung trung binh lam moc duoi con P99
                        # lam tran thi voi phan phoi lech phai nang, trung
                        # binh CO THE vuot P99 (gap that o Online Boutique:
                        # recommendationservice_latency-50 co mean=0.010 >
                        # P99=0.0077) va bien tinh ra am.
                        'p50': float(np.percentile(vals, 50)),
                        'p90': float(np.percentile(vals, 90)),
                        'p95': float(np.percentile(vals, 95)),
                        'p99': float(np.percentile(vals, 99)),
                    }
        self._ood_stats_cache = stats
        return stats

    def _classify_ood_confidence(self, samples: pd.DataFrame, injection_node: str = None) -> Tuple[str, List[str]]:
        """Compare each DOWNSTREAM node's projected mean against ITS OWN
        training percentiles. Returns (worst_confidence, flagged_node_labels).
        `samples` is the DataFrame `simulate_intervention` already computed
        via `gcm.interventional_samples` -- no extra SCM call is made here.

        `injection_node` is excluded from the check by construction: a do(x)
        query is meant to explore values beyond what the injected node has
        historically taken, so flagging it against its OWN training envelope
        is close to tautological and carries no information about whether an
        UNINTENDED downstream effect occurred -- exactly the distinction this
        guard's own description (Section "Workload Propagation" in the paper)
        draws. An earlier version of this method omitted this exclusion and
        flagged the injection node on effectively every non-trivial delta,
        discovered when 150/150 real RQ3-parsed cases came back 'low'
        (data/processed/scm_results/g7_rq3_scale_evaluation.csv) -- see
        Section "Discussion" for the corrected, still-imperfect result."""
        stats = self._ood_train_stats()
        worst = 'high'
        flagged = []
        for node in samples.columns:
            if node not in stats or node == injection_node:
                continue
            s = stats[node]
            val = float(samples[node].mean())
            if val <= s['p90']:
                conf = 'high'
            elif val <= s['p95']:
                conf = 'medium'
            elif val <= s['p99'] or val <= 2 * s['max']:
                conf = 'low'
            else:
                conf = 'very_low'
            if conf != 'high':
                flagged.append(f"{node} (P95={s['p95']:.2f}, projected={val:.2f})")
            if self._OOD_CONF_ORDER[conf] > self._OOD_CONF_ORDER[worst]:
                worst = conf
        return worst, flagged

    def simulate_intervention(
        self,
        injection_service: str = None,
        delta_pct: float = 25.0,
        n_samples: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """Tương thích SimulationAgent.simulate_intervention().

        Dùng deterministic_forward() (Proposition 2 machinery, xem
        certified_envelope) thay vì gcm.interventional_samples(): can thiệp
        trong hệ này luôn xảy ra tại gateway (in-degree=0), nên do(X)=
        condition(X) đồng nhất (RQ5b) -- xác nhận thực nghiệm trên cả 2 hệ
        thống (SockShop, Train Ticket) rằng deterministic forward khớp
        median <1% với trung bình Monte Carlo, nhanh hơn ~14x, và không còn
        nhiễu lấy mẫu từng khiến G7 trả về confidence KHÁC NHAU giữa hai lần
        gọi với CÙNG input (bug đã tái lập: độ lệch tương đối giữa các lần
        gọi lặp lại tới 36% ở node xấu nhất, dù model/delta không đổi).
        simulate_joint_intervention() đã dùng đúng cách này từ trước; hàm
        đơn-can-thiệp này trước đó là ngoại lệ còn sót lại dùng gcm.
        n_samples giữ trên chữ ký để tương thích ngược, không còn dùng.
        """
        # Reset G7 state up front so an early return below (no model / unknown
        # column) can never leak a stale confidence level from a PRIOR call.
        self._last_ood_confidence = "high"
        self._last_ood_flagged = []

        if injection_service is None:
            injection_service = self.default_injection

        if self.global_dag_model is None or self.df_baseline is None:
            return {}

        injection_col = f"{injection_service}_workload"
        if injection_col not in self.df_baseline.columns:
            return {}

        base_wl   = float(self.df_baseline[injection_col].mean())
        target_wl = base_wl * (1.0 + delta_pct / 100.0)

        print(f"  [Simulation] do({injection_col} = {base_wl:.2f} -> {target_wl:.2f} req/s, +{delta_pct}%)")

        vals, breached = deterministic_forward(
            self.dag_graph, self.global_dag_model, self.df_baseline,
            {injection_col: target_wl})

        # G7: _classify_ood_confidence() chỉ đọc samples[node].mean() --
        # bọc vals (dict) thành DataFrame 1 dòng để tái dùng nguyên hàm đó,
        # không mất thông tin gì so với trước (Monte Carlo cũng chỉ bị đọc
        # qua .mean(), chưa từng dùng phương sai).
        samples = pd.DataFrame([vals])
        self._last_ood_confidence, self._last_ood_flagged = self._classify_ood_confidence(
            samples, injection_node=injection_col)

        hops = self._compute_hops(injection_service)

        results = {}
        for svc in self.services:
            entry = {'n_hops': hops.get(svc, -1)}
            for metric_key, col_suffix in [
                ('cpu_change_pct',     '_cpu'),
                ('mem_change_pct',     '_mem'),
                ('latency_change_pct', '_latency-50'),
            ]:
                col = f"{svc}{col_suffix}"
                base_key = metric_key.replace('_change_pct', '')
                if col in self.df_baseline.columns and col in vals:
                    base = float(self.df_baseline[col].mean())
                    if col in breached:
                        entry[metric_key] = float('inf')
                        entry[f"{base_key}_predicted"] = float('inf')
                    else:
                        pred = vals[col]
                        chg  = (pred - base) / abs(base) * 100.0 if base != 0 else 0.0
                        entry[metric_key] = round(chg, 1)
                        entry[f"{base_key}_predicted"] = round(pred, 2)
                    entry[f"{base_key}_baseline"]  = round(base, 2)
                    # Do tin cay cua node nay (_node_confidence(), dua tren
                    # evaluate_node_stability() rieng cua node) -- danh dau ngay canh so, khong bat
                    # nguoi doc phai tu tra select_key_nodes() rieng.
                    entry[f"{base_key}_confidence"] = self._node_confidence(col)
                else:
                    entry[metric_key] = None
                    entry[f"{base_key}_confidence"] = 'no_dag_node'
            results[svc] = entry

        return results

    def simulate_joint_intervention(
        self,
        injection_services: List[str] = None,
        delta_pct: float = 25.0,
    ) -> Dict[str, Any]:
        """Nhu simulate_intervention(), nhung can thiep DONG THOI tai NHIEU
        gateway -- truoc day chi co o muc chung minh (joint_certified_
        envelope, Proposition 3), chua lo ra o lop API chinh nguoi dung/
        assess_capacity() hay dung.

        injection_services: mac dinh self.gateways (TOAN BO gateway hop le
        suy tu graph, xem __init__) -- khong con gioi han 1 gateway "chinh"
        duy nhat nhu simulate_intervention().

        Dung deterministic_forward() (KHONG Monte Carlo, xem module
        deterministic_forward.py) thay vi gcm.interventional_samples() --
        nhat quan voi huong toi uu phep do() da chon xuyen suot phien lam
        viec nay, va tranh sai so chon mau khi can thiep nhieu diem cung
        luc (moi diem them mot nguon nhieu ngau nhien neu dung Monte Carlo).

        Tra ve dict CUNG SCHEMA voi simulate_intervention() (moi service:
        cpu/mem/latency_change_pct + _predicted/_baseline/_confidence),
        chi khac o cho hieu ung la TONG HOP cua tat ca gateway can thiep
        cung luc (khong phai cong don tung gateway rieng le -- xem
        docstring joint_certified_envelope ve ly do khong duoc cong don
        cho node latency phi tuyen)."""
        self._last_ood_confidence = "high"
        self._last_ood_flagged = []

        if injection_services is None:
            injection_services = self.gateways or (
                [self.default_injection] if self.default_injection else [])

        if self.global_dag_model is None or self.df_baseline is None or not injection_services:
            return {}

        injections = {}
        for svc in injection_services:
            col = f"{svc}_workload"
            if col not in self.df_baseline.columns:
                continue
            base_wl = float(self.df_baseline[col].mean())
            injections[col] = base_wl * (1.0 + delta_pct / 100.0)

        if not injections:
            return {}

        print(f"  [Joint Simulation] do({list(injections.keys())} = +{delta_pct}%, "
              f"{len(injections)} gateway dong thoi)")

        vals, breached = deterministic_forward(self.dag_graph, self.global_dag_model,
                                                self.df_baseline, injections)

        results = {}
        for svc in self.services:
            entry = {}
            for metric_key, col_suffix in [
                ('cpu_change_pct',     '_cpu'),
                ('mem_change_pct',     '_mem'),
                ('latency_change_pct', '_latency-50'),
            ]:
                col = f"{svc}{col_suffix}"
                base_key = metric_key.replace('_change_pct', '')
                if col in self.df_baseline.columns and col in vals:
                    base = float(self.df_baseline[col].mean())
                    if col in breached:
                        entry[metric_key] = float('inf')
                        entry[f"{base_key}_predicted"] = float('inf')
                    else:
                        pred = vals[col]
                        chg = (pred - base) / abs(base) * 100.0 if base != 0 else 0.0
                        entry[metric_key] = round(chg, 1)
                        entry[f"{base_key}_predicted"] = round(pred, 2)
                    entry[f"{base_key}_baseline"] = round(base, 2)
                    entry[f"{base_key}_confidence"] = self._node_confidence(col)
                else:
                    entry[metric_key] = None
                    entry[f"{base_key}_confidence"] = 'no_dag_node'
            results[svc] = entry

        return results

    # ----------------------------------------------------------
    # PROPOSITION 2: CERTIFIED CAPACITY ENVELOPE
    # ----------------------------------------------------------
    # Layer 3 (parser_agent.py) guarantees the emitted delta lies in
    # [5,50] (Proposition 1(ii)). Because every Tier-1/Tier-2 mechanism
    # is fit with non-negative coefficients (LinearRegression(positive=
    # True)) and the queueing transform phi(w)=w/(c-w) is monotone
    # increasing on w<c, the composition of the whole DAG is monotone
    # non-decreasing in delta (proof: induction over topological order,
    # composition of monotone non-decreasing functions is monotone
    # non-decreasing). Two deterministic forward passes therefore bound
    # every downstream node's POINT ESTIMATE for every delta* in [5,50]
    # Layer 3 could emit -- not a probabilistic bound on the realised
    # future value (noise terms N_j are not propagated here; see
    # docstring of certified_envelope for what this does and does not
    # certify).
    #
    # This must NOT be implemented via simulate_intervention() / gcm.
    # interventional_samples(): that draws stochastic Monte Carlo noise
    # per node, which reintroduces exactly the ambiguity a certificate
    # is meant to remove (confirmed empirically: two live calls at the
    # same delta gave CPU predictions that differed only due to sampling
    # noise, with no change to the fitted mechanism at all).

    def _deterministic_forward(
        self, injection_node: str, target_value: float
    ) -> Tuple[Dict[str, float], List[str]]:
        """Noise-free point-estimate propagation: walk the DAG in
        topological order, call each fitted mechanism's own sklearn
        .predict() on its parents' point estimates (no additive-noise
        sampling). Returns (values, breached) where `breached` lists
        latency nodes whose parent workload reached or exceeded that
        node's own fitted capacity ceiling (phi's domain boundary) --
        flagged explicitly rather than silently extrapolated through a
        diverging queueing transform.

        Thin wrapper over src/scm/deterministic_forward.py (shared with
        _deterministic_forward_multi below and with node_impact.py's
        rank_user_impact) -- single injection is just the multi-injection
        case with one key."""
        return deterministic_forward(
            self.dag_graph, self.global_dag_model, self.global_df_baseline,
            {injection_node: target_value})

    def _check_monotone_precondition(self) -> List[str]:
        """Runtime check of Proposition 2's precondition (a): every
        fitted mechanism coefficient is non-negative. Re-asserted here
        rather than only trusted from the solver constraint, because
        RQ3's own finding (a hard-coded fallback silently violating
        Proposition 1) is precisely the failure mode this guards
        against for Proposition 2: a design that is correct is not
        thereby an implementation that is correct."""
        violations = []
        for node in self.dag_graph.nodes():
            if self.dag_graph.in_degree(node) == 0:
                continue
            sk = self.global_dag_model.causal_mechanism(node).prediction_model.sklearn_model
            coef = sk.model_.coef_ if isinstance(sk, QueueingLatencyRegressor) else getattr(sk, 'coef_', None)
            if coef is not None and bool((np.array(coef) < 0).any()):
                violations.append(node)
        return violations

    def _check_monotone_numeric(self, n_diem: int = 60, le: float = 1.5,
                                dung_sai: float = 1e-9) -> List[Dict[str, Any]]:
        """Kiem tien dieu kien (a) cua Menh de 2 BANG SO, khong qua he so.

        Vi sao can them, khi da co `_check_monotone_precondition`: ham do chi doc
        `coef_ >= 0`. Mot co che co the co MOI he so khong am ma VAN khong don dieu
        -- vi du mot co che chuyen nhanh theo dieu kien. Da gap that: ban dau
        `CoCheCongDai` duoc cai kieu "ngoai dai thi bo cha tai nguyen", co
        `coef_ >= 0` nen qua duoc phep kiem he so, nhung du bao TUT 7,09 ngay tai
        bien dai -- tuc pha dung tien dieu kien ma phep kiem do ton tai de bao ve.
        (Da sua thanh KEP; xem src/scm/co_che_cong_dai.py. Tu 2026-10-10,
        `CoCheCongDai` khong con duoc goi trong file nay -- xem _assign_dag_mechanisms --
        nen phep kiem nay hien qua voi moi node vi tat ca deu LinearRegression
        don dieu theo thiet ke; van giu lam luoi an toan so.)

        Phep kiem: voi moi node co cha, quet TUNG cha tu min den `le` x max tren du
        lieu da fit, giu cac cha khac o trung vi, roi doi chieu du bao co don dieu
        khong giam. Quet VUOT qua max vi Menh de 2 duoc danh gia tren
        delta in [5%, 50%] -- vung ma cha ra ngoai dai hieu chuan la chuyen thuong.

        Node `_latency-50` duoc bao cao RIENG: `QueueingLatencyRegressor` la
        W/(C-W), don dieu CHI khi W < C -- dung tien dieu kien (b) cua Menh de 2,
        khong phai (a). Vuot C thi no phan ky theo thiet ke, khong phai loi.

        Tra ve danh sach vi pham, moi muc gom node, cha, cu tut lon nhat.
        """
        if getattr(self, 'global_df_baseline', None) is None:
            return []
        df = self.global_df_baseline
        vi_pham = []
        for node in self.dag_graph.nodes():
            if self.dag_graph.in_degree(node) == 0 or node not in df.columns:
                continue
            cha = sorted(self.dag_graph.predecessors(node))
            if not all(c in df.columns for c in cha):
                continue
            sk = self.global_dag_model.causal_mechanism(node).prediction_model.sklearn_model
            goc = np.array([[float(df[c].median()) for c in cha]])
            for i, c in enumerate(cha):
                lo, hi = float(df[c].min()), float(df[c].max())
                if not np.isfinite([lo, hi]).all() or hi <= lo:
                    continue
                X = np.repeat(goc, n_diem, axis=0)
                X[:, i] = np.linspace(lo, hi * le if hi > 0 else hi, n_diem)
                try:
                    y = np.asarray(sk.predict(X), dtype=float).ravel()
                except Exception:
                    continue                      # co che khong nhan duoc khung nay
                d = np.diff(y)
                if (d < -dung_sai).any():
                    vi_pham.append({
                        'node': node, 'cha': c, 'cu_tut_lon_nhat': float(d.min()),
                        'la_latency': node.endswith('_latency-50'),
                    })
        return vi_pham

    def certified_envelope(
        self,
        injection_service: str = None,
        delta_min_pct: float = 5.0,
        delta_max_pct: float = 50.0,
    ) -> Dict[str, Any]:
        """Proposition 2. Two deterministic forward passes at the
        verified boundary of the intervention delta certify, for EVERY
        delta* in [delta_min_pct, delta_max_pct], a coordinate-wise
        interval containing the point-estimate (conditional-mean, zero-
        noise) value of every downstream node. This is a guarantee about
        the fitted deterministic mechanism, NOT a probabilistic
        prediction interval on the realised future observation: it does
        not account for N_j, nor for mechanism misspecification. See the
        existing G7 OOD-confidence guard (_classify_ood_confidence) for
        the complementary, separate signal of whether a node's projected
        value falls outside its own training distribution -- report both
        together, they answer different questions.

        Raises RuntimeError if the monotonicity precondition does not
        hold on the currently fitted model (see _check_monotone_
        precondition) -- the certificate must never be issued silently
        over a violated precondition.
        """
        violations = self._check_monotone_precondition()
        if violations:
            raise RuntimeError(
                f"Proposition 2 precondition violated (beta<0) at: {violations}. "
                "Certificate withheld -- fix the mechanism before certifying."
            )
        # Phep kiem BANG SO: bat duoc co che co he so khong am ma van khong don
        # dieu (vd chuyen nhanh theo dieu kien) -- loai vi pham ma phep kiem he so
        # tren KHONG thay. Node latency bo qua o day: chung chiu tien dieu kien (b).
        so = [v for v in self._check_monotone_numeric() if not v['la_latency']]
        if so:
            raise RuntimeError(
                f"Proposition 2 precondition violated NUMERICALLY (khong don dieu du "
                f"he so khong am) at: {[(v['node'], v['cha'], round(v['cu_tut_lon_nhat'], 4)) for v in so]}. "
                "Certificate withheld."
            )

        if injection_service is None:
            injection_service = self.default_injection
        inj_col = f"{injection_service}_workload"
        if inj_col not in self.global_df_baseline.columns:
            return {}

        base_wl = float(self.global_df_baseline[inj_col].mean())
        w_lo = base_wl * (1.0 + delta_min_pct / 100.0)
        w_hi = base_wl * (1.0 + delta_max_pct / 100.0)

        # TIEN DIEU KIEN (b) cua Menh de 2: Y_PA_j(delta) < c_j tren TOAN khoang.
        #
        # Kiem o DUNG HAI DAU mut la DU -- nhung chi vi (a) da duoc kiem o tren.
        # (a) cho don dieu, nen max cua Y_PA_j tren [delta_min, delta_max] dat tai
        # delta_max; hai lan quet tat dinh do dat hai bien la bao duoc ca khoang.
        # PHU THUOC NAY LA THAT: neu (a) khong duoc kiem (hoac bi kiem bang mot
        # dieu kien DU nhu `coef_ >= 0`, vi pham van lot -- xem
        # _check_monotone_numeric), thi phep kiem hai dau mut KHONG con bao duoc
        # khoang giua, va chung nhan co the duoc cap tren mot tien dieu kien da vo
        # o giua khoang. Vi vay thu tu goi o ham nay la BAT BUOC: (a) truoc, (b) sau.
        #
        # Node vi pham nhan 'CEILING_BREACHED' thay vi mot khoang -- tu choi chung
        # nhan RIENG node do. Can thiet vi QueueingLatencyRegressor TU KEP
        # (`X_capped = min(X, capacity_*0.99)`), nen vuot tran no khong bao loi ma
        # BAO HOA IM LANG -- du bao do tre phang ra, tuc danh gia THAP nguy co dung
        # vung tai cao. Da kiem: tai delta_max = 50% khong node nao vi pham; tu 150%
        # co 5 node latency bi tu choi, tu 600% ca 7/7.
        Y_lo, breached_lo = self._deterministic_forward(inj_col, w_lo)
        Y_hi, breached_hi = self._deterministic_forward(inj_col, w_hi)
        breached = set(breached_lo) | set(breached_hi)

        envelope: Dict[str, Any] = {}
        for node in self.dag_graph.nodes():
            if node == inj_col:
                continue
            if node in breached:
                envelope[node] = 'CEILING_BREACHED'
                continue
            lo, hi = Y_lo.get(node), Y_hi.get(node)
            if lo is None or hi is None:
                continue
            # min/max rather than assuming lo<=hi: a cheap defensive net,
            # not a substitute for the precondition check above.
            envelope[node] = {'lo': min(lo, hi), 'hi': max(lo, hi)}
        return envelope

    # ----------------------------------------------------------
    # PROPOSITION 3: JOINT INTERVENTION ENVELOPE
    # ----------------------------------------------------------
    # Two requirements approved in the same window intervene at two
    # (possibly different) nodes simultaneously. Proposition 2's
    # monotonicity argument extends unchanged to a vector of
    # simultaneous injections -- the induction over topological order
    # only ever used that each mechanism is monotone non-decreasing in
    # EACH of its parents, which holds regardless of how many root
    # variables feed the graph. So the joint envelope over K simultaneous
    # deltas still costs exactly two deterministic forward passes: all K
    # injections at their delta_min together, all K at their delta_max
    # together.
    #
    # What does NOT carry over is additivity. Tier-1/Tier-2 CPU/mem/
    # workload mechanisms are affine, so their joint effect equals the
    # exact sum of each intervention's marginal effect computed alone --
    # superposition holds. The latency mechanism is not affine: phi(w) =
    # w/(c-w) is convex increasing on w<c, so for any node whose parent
    # workload receives contributions from two converging interventions,
    # Jensen's inequality gives
    #     phi(w0+d1+d2) - phi(w0)  >=  [phi(w0+d1)-phi(w0)] + [phi(w0+d2)-phi(w0)],
    # i.e. the TRUE joint latency increase is never less than the sum of
    # the two increases computed separately -- strictly greater whenever
    # both d1,d2>0 and phi is strictly convex there. Checking two
    # requirements individually against a shared latency bottleneck is
    # therefore necessary but not sufficient: the joint envelope must be
    # computed with both injections active, not approximated by summing
    # two single-intervention envelopes.

    def _deterministic_forward_multi(
        self, injections: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """As _deterministic_forward, but accepts several simultaneous
        do(node=value) injections (keys are e.g. 'front-end_workload',
        'orders_workload'). Nodes not listed are computed from their
        parents as usual; injected nodes take their given value directly,
        overriding whatever their own mechanism would have produced.

        Thin wrapper over src/scm/deterministic_forward.py -- see
        _deterministic_forward above."""
        return deterministic_forward(
            self.dag_graph, self.global_dag_model, self.global_df_baseline, injections)

    def joint_certified_envelope(
        self,
        injection_services: List[str],
        delta_min_pct: float = 5.0,
        delta_max_pct: float = 50.0,
    ) -> Dict[str, Any]:
        """Proposition 3. Certifies an interval for every downstream node
        under K simultaneous interventions, for every combination of
        delta*_1,...,delta*_K each in [delta_min_pct, delta_max_pct] --
        still exactly two deterministic forward passes regardless of K,
        by the same monotonicity argument as certified_envelope(). See
        the module-level note above on why this must NOT be approximated
        by summing K single-intervention envelopes for any node whose
        latency depends on a workload with contributions from more than
        one of the injected services."""
        violations = self._check_monotone_precondition()
        if violations:
            raise RuntimeError(
                f"Proposition 2/3 precondition violated (beta<0) at: {violations}. "
                "Certificate withheld."
            )
        inj_cols = [f"{s}_workload" for s in injection_services]
        for c in inj_cols:
            if c not in self.global_df_baseline.columns:
                return {}
        bases = {c: float(self.global_df_baseline[c].mean()) for c in inj_cols}

        lo_injections = {c: bases[c] * (1.0 + delta_min_pct / 100.0) for c in inj_cols}
        hi_injections = {c: bases[c] * (1.0 + delta_max_pct / 100.0) for c in inj_cols}
        Y_lo, breached_lo = self._deterministic_forward_multi(lo_injections)
        Y_hi, breached_hi = self._deterministic_forward_multi(hi_injections)
        breached = set(breached_lo) | set(breached_hi)

        envelope: Dict[str, Any] = {}
        for node in self.dag_graph.nodes():
            if node in inj_cols:
                continue
            if node in breached:
                envelope[node] = 'CEILING_BREACHED'
                continue
            lo, hi = Y_lo.get(node), Y_hi.get(node)
            if lo is None or hi is None:
                continue
            envelope[node] = {'lo': min(lo, hi), 'hi': max(lo, hi)}
        return envelope

    def compare_joint_vs_naive_sum(
        self, injection_services: List[str], delta_pct: float = 50.0
    ) -> Dict[str, Dict[str, float]]:
        """Empirical companion to Proposition 3: computes, for a single
        fixed delta applied to each of K services simultaneously, (a) the
        TRUE joint point estimate (both/all injections active in one
        deterministic forward pass) versus (b) the NAIVE sum (each
        injection's marginal delta computed alone against baseline, then
        added). Returns, per downstream node, {'joint':..., 'naive_sum':
        ..., 'gap': joint-naive_sum}. Proposition 3 predicts gap==0 (up to
        floating point) for every affine Tier-1/Tier-2 node, and gap>=0,
        strictly >0 where a latency node's parent workload receives a
        genuine contribution from more than one injected service."""
        inj_cols = [f"{s}_workload" for s in injection_services]
        bases = {c: float(self.global_df_baseline[c].mean()) for c in inj_cols}
        target = {c: bases[c] * (1.0 + delta_pct / 100.0) for c in inj_cols}

        baseline_vals, _ = self._deterministic_forward_multi({})
        joint_vals, _ = self._deterministic_forward_multi(target)

        # Naive sum: baseline + each injection's own marginal delta, added.
        marginals = {c: self._deterministic_forward_multi({c: target[c]})[0] for c in inj_cols}
        naive_sum: Dict[str, float] = {}
        for node, base_v in baseline_vals.items():
            if node in inj_cols:
                continue
            total = base_v
            for c in inj_cols:
                total += marginals[c].get(node, base_v) - base_v
            naive_sum[node] = total

        out: Dict[str, Dict[str, float]] = {}
        for node in joint_vals:
            if node in inj_cols or node not in naive_sum:
                continue
            j, n = joint_vals[node], naive_sum[node]
            out[node] = {'joint': j, 'naive_sum': n, 'gap': j - n}
        return out

    def get_dag_summary(self) -> Dict[str, Any]:
        if self.dag_graph is None:
            return {'n_nodes': 0, 'n_edges': 0}
        return {'n_nodes': len(self.dag_graph.nodes()), 'n_edges': len(self.dag_graph.edges())}

    def get_baseline_workload(self, service: str) -> float:
        if self.global_df_baseline is not None:
            node = f"{service}_workload"
            if node in self.global_df_baseline:
                return float(self.global_df_baseline[node])
        return 24.0

    # ----------------------------------------------------------
    # NODE IMPACT / STABILITY RANKING (src/scm/node_impact.py)
    # ----------------------------------------------------------
    # "Node nao anh huong nguoi dung nhat" va "node nao du bao on dinh
    # nhat" la 2 cau hoi khac nhau tra loi boi node_impact.py (xem
    # docstring module do). Cac method duoi day CHI truyen dung state cua
    # CHINH agent nay (dag_graph/global_dag_model/global_df_baseline da
    # fit boi train_accurate_path) vao 2 ham system-agnostic do -- khong
    # lap lai logic.

    def rank_user_impact(
        self, target_nodes: List[str] = None, gateway: str = None, delta_pct: float = 20.0
    ) -> pd.DataFrame:
        """Xep hang tung cap (node, target) trong Global DAG theo
        elasticity doi voi trai nghiem nguoi dung (mac dinh target_nodes:
        TOAN BO node '..._latency-50' trong do thi -- xem node_impact.
        rank_user_impact() de biet vi sao mot target duy nhat (vd chi
        latency cua gateway) cho ket qua suy bien tren DAG hien tai)."""
        if self.dag_graph is None or self.global_dag_model is None:
            return pd.DataFrame()
        return _rank_user_impact(self.dag_graph, self.global_dag_model,
                                  self.global_df_baseline, target_nodes=target_nodes,
                                  gateway=gateway, delta_pct=delta_pct)

    def evaluate_node_stability(self) -> pd.DataFrame:
        """Danh gia do on dinh du bao (MAPE/R2 held-out, protocol OOD Gold
        Standard) cua CHINH co che DA FIT (khong refit ban sao moi -- xem
        node_impact.evaluate_node_stability() ve ly do quan trong cua dieu
        nay) cho tung node co cha trong Global DAG."""
        if self.dag_graph is None or self.global_dag_model is None or self.global_df_baseline is None:
            return pd.DataFrame()
        return _evaluate_node_stability(self.dag_graph, self.global_dag_model, self.global_df_baseline)

    def select_key_nodes(
        self, target_nodes: List[str] = None, gateway: str = None, delta_pct: float = 20.0,
    ) -> Dict[str, List[str]]:
        """Gop rank_user_impact() + evaluate_node_stability() thanh khuyen
        nghi node nao nen uu tien du bao/theo doi: 'recommended' (anh
        huong nguoi dung manh VA du bao on dinh, hoac la node goc doc truc
        tiep tu telemetry), 'unstable_but_impactful' (anh huong manh nhung
        du bao KHONG dang tin -- canh bao rieng, khong am tham bo qua). Xem
        node_impact.select_key_nodes()."""
        impact_df = self.rank_user_impact(target_nodes, gateway, delta_pct)
        stability_df = self.evaluate_node_stability()
        root_nodes = {n for n in self.dag_graph.nodes() if self.dag_graph.in_degree(n) == 0}
        return _select_key_nodes(impact_df, stability_df, root_nodes=root_nodes)

    def _get_key_nodes(self) -> Dict[str, List[str]]:
        """select_key_nodes() voi tham so mac dinh, cache lai -- CHI dung
        cho cau hoi "node nao nen theo doi o muc TOAN HE THONG" (vd bao
        cao tong quan). KHONG dung cho _node_confidence() (xem do): mot
        node CapacityAgent DA QUYET DINH bao cao (vi request/service dang
        hoi toi no) can duoc danh gia do tin cay CUA RIENG NO, khong phai
        bi che boi cau hoi "no co nam trong top-impact TOAN CUC hay
        khong" -- 2 cau hoi khac nhau. Vi du da phat hien tren Train
        Ticket: target mac dinh (latency) khien MOI node co elasticity=0
        (README #7), nen select_key_nodes() tra ve rong cho MOI node --
        neu _node_confidence() dung cache nay, no se bao 'not_ranked' cho
        ca 28 node latency hoi hong, che mat dung van de can bao."""
        if self._key_nodes_cache is None:
            self._key_nodes_cache = self.select_key_nodes() if self.dag_graph is not None else {
                'high_impact': [], 'recommended': [], 'unstable_but_impactful': []}
        return self._key_nodes_cache

    def _get_stability_df(self) -> pd.DataFrame:
        """evaluate_node_stability() cache lai (mot lan/train(), xem
        _get_key_nodes() ve ly do cache theo train() chu khong theo lan
        goi) -- day la nguon THAT cho _node_confidence(), khong phai
        select_key_nodes()."""
        if self._stability_cache is None:
            self._stability_cache = self.evaluate_node_stability() if self.dag_graph is not None else pd.DataFrame()
        return self._stability_cache

    def mark_node_unreliable(self, node: str, reason: str = None):
        """Danh dau TAY mot node la khong dang tin, doc lap voi select_
        key_nodes() tu dong -- dung khi biet truoc mot van de KHONG the
        (hoac chua) bat duoc qua elasticity/stability tu dong, vd:
          - G7 da ghi nhan catalogue_cpu tren Sock Shop lech phan phoi
            nang, bao dong gia (README "Gioi han da biet" #4).
          - README #7: ca 28 node latency cua Train Ticket co he so = 0
            (evaluate_node_stability() da bat duoc dieu nay qua kiem coef_
            truc tiep -- MAPE mot minh KHONG bat duoc, xem
            node_impact._is_degenerate_mechanism -- nhung mot nguoi dung
            co the muon danh dau ngay ca khi chua chay lai danh gia).
        Danh dau nay LUON duoc _node_confidence() uu tien hon ket qua tu
        dong (xem do)."""
        self._manual_unreliable_nodes[node] = reason or 'flagged_manually'

    def unmark_node_unreliable(self, node: str):
        """Bo danh dau TAY (neu co) -- khong loi neu node chua duoc danh dau."""
        self._manual_unreliable_nodes.pop(node, None)

    def _node_confidence(self, node: str) -> str:
        """Do tin cay CUA RIENG node nay (khong phu thuoc no co "quan
        trong toan cuc" hay khong -- xem _get_key_nodes() ve ly do TACH
        rieng khoi select_key_nodes()/rank_user_impact()). Uu tien theo
        thu tu:
          1. 'manually_flagged' : da mark_node_unreliable() cho node nay
                                   -- LUON thang, bat ke danh gia tu dong.
          2. 'no_dag_node'      : node khong ton tai trong Global DAG (vd
                                   Socket -- Tier-2 khong co canh workload
                                   ->socket) -- khong co gi de danh gia.
          3. 'stable'           : node GOC (doc truc tiep tu telemetry,
                                   khong qua co che du bao nao -- vd
                                   workload cua gateway), HOAC co
                                   evaluate_node_stability() voi tag
                                   EXCELLENT/FAIR.
          4. 'unstable'         : co evaluate_node_stability() voi tag
                                   POOR -- day la nguyen nhan README #7
                                   (28 node latency Train Ticket) duoc
                                   bat o day, KHONG phai qua top-impact.
          5. 'unknown'          : co co che (khong phai node goc) nhung
                                   khong du du lieu OOD held-out de danh
                                   gia (vd < min_rows) -- khac 'stable',
                                   khong nen ngam dinh la dang tin."""
        if node in self._manual_unreliable_nodes:
            return 'manually_flagged'
        if self.dag_graph is None or node not in self.dag_graph.nodes():
            return 'no_dag_node'
        if self.dag_graph.in_degree(node) == 0:
            return 'stable'
        stability_df = self._get_stability_df()
        row = stability_df[stability_df['node'] == node] if not stability_df.empty else stability_df
        if row.empty:
            return 'unknown'
        tag = row.iloc[0]['tag']
        return 'unstable' if tag == 'POOR' else 'stable'

    # ----------------------------------------------------------
    # CHU TRÌNH REACT TOÀN PHẦN (THE CORE AGENTIC METHOD)
    # ----------------------------------------------------------
    # BIEN DU PHONG (HEADROOM) -- DAI LUONG DAU RA CHINH
    # ----------------------------------------------------------
    # Nguong CHINH SACH (policy), KHONG phai do duoc tu du lieu -- khai bao
    # tuong minh o day thay vi chon tay trong ham, de doi duoc mot cho.
    HEADROOM_WARNING_FRAC = 0.50    # tieu >50% bien con lai -> WARNING
    HEADROOM_CRITICAL_FRAC = 1.00   # vuot qua tran da quan sat -> CRITICAL
    # Tran P99 duoc uoc luong tu mau huu han nen CHINH NO co sai so. Thay vi
    # dat them mot nguong tuy y de loai node "duoi nang" (ban dau o day tung
    # co CEILING_TAIL_RATIO_MAX=10.0, chon sau khi thay catalogue_cpu o 210x
    # -- tuc gan nguong theo MOT quan sat tren MOT he; do phan bo max/P99 tren
    # ca 2 he cho thay phan bo lien tuc, P50=1.67 / P90=5.2 / P95=26.3, khong
    # co vach ngan tu nhien nao tai 10), ta BOOTSTRAP tran va de trang thai
    # tu bao "UNDECIDED" khi khoang tin cay vat qua ranh gioi quyet dinh.
    # Khong them hang so nao: ranh gioi van la hai nguong chinh sach o tren.
    CEILING_BOOTSTRAP_B = 200
    CEILING_BOOTSTRAP_N = 2000      # tran mau moi lan resample (du on dinh, du nhanh)
    CEILING_CI_LOW, CEILING_CI_HIGH = 5, 95

    def compute_headroom(self, sim_result: Dict[str, Any]) -> Dict[str, Any]:
        """Quy doi du bao sang BIEN DU PHONG: tinh nang nay an bao nhieu phan
        tram khoang cach con lai tu muc hien tai toi tran van hanh.

        Vi sao doi dau ra: tran nang luc vat ly C KHONG dinh danh duoc tu du
        lieu chua bao hoa (do thuc nghiem tren Alibaba: phi(W)=W/(C-W) suy
        bien, moi co che latency co rang buoc bi ep he so ve 0). Nen thay vi
        hua mot con so latency khong giu duoc, bao cao dai luong CO tin hieu
        that va don dieu: do chiem dung tai nguyen so voi bien con lai.

        Tran = P99 cua CHINH node do trong du lieu van hanh binh thuong --
        "muc cao nhat da thuc su quan sat thay he chay duoc", khong phai mot
        nguong % viet cung (CPU trong bo du lieu nay la core/byte, khong phai
        phan tram -- nguong `>70.0` trong ban assess_capacity truoc day vi the
        gan nhu khong bao gio kich hoat).

        MOI metric deu duoc xu ly GIONG NHAU -- khong loai cung metric nao.
        Mot metric duoc bao 'not_forecastable' khi CO CHE CUA CHINH NODE DO
        tren CHINH he thong nay bi danh gia la khong dang tin
        (_node_confidence() tra 'unstable'/'manually_flagged'/'unknown'), chu
        khong phai vi mot ket luan viet tay. Nho vay: tren SockShop/Train
        Ticket hau het node latency se tu bi loai (README #7: 28/28 co che
        latency Train Ticket fit ra coef=0, evaluate_node_stability() gan tag
        POOR), con mot he thong tuong lai co latency that su don dieu thi VAN
        duoc bao cao binh thuong -- quy tac do duoc, khong phai gia dinh.

        Tra ve {service: {metric: {...}}} voi moi metric gom:
          baseline, projected, ceiling_p99 (+ khoang tin cay bootstrap),
          headroom_before, consumed_pct va [consumed_lo_pct, consumed_hi_pct],
          remaining_pct, status.
        """
        stats = self._ood_train_stats()
        out: Dict[str, Any] = {}
        for svc, entry in (sim_result or {}).items():
            node_out = {}
            for metric_key, col_suffix in (('cpu', '_cpu'), ('mem', '_mem'),
                                            ('latency', '_latency-50')):
                col = f"{svc}{col_suffix}"
                base = entry.get(f'{metric_key}_baseline')
                proj = entry.get(f'{metric_key}_predicted')
                s = stats.get(col)
                if base is None or proj is None or s is None:
                    continue

                # Cong do duoc, ap cho MOI metric nhu nhau: co che cua node nay
                # tren he nay co dang tin khong?
                conf = self._node_confidence(col)
                if conf != 'stable':
                    node_out[metric_key] = {
                        'status': 'not_forecastable', 'confidence': conf,
                        'reason': f"co che cua node nay duoc danh gia '{conf}' "
                                   f"(evaluate_node_stability/mark_node_unreliable)"}
                    continue
                if not (np.isfinite(base) and np.isfinite(proj)):
                    node_out[metric_key] = {'status': 'unavailable',
                                            'reason': 'du bao khong huu han (ceiling breached)'}
                    continue

                ceiling = float(s['p99'])
                floor = float(s['p50'])
                lo_c, hi_c = self._ceiling_ci(col)
                rec = {'baseline': round(float(base), 4),
                       'projected': round(float(proj), 4),
                       'floor_p50': round(floor, 4),
                       'ceiling_p99': round(ceiling, 4),
                       'ceiling_ci': [round(lo_c, 4), round(hi_c, 4)],
                       'confidence': conf}

                # BIEN = khoang van hanh quan sat duoc P50 -> P99 (hai PHAN VI
                # cua CUNG phan phoi), KHONG phai P99 - trung binh: trung binh
                # va phan vi la hai ho thong ke khac nhau, va voi phan phoi
                # lech phai nang trung binh co the VUOT P99 -> bien am (gap
                # that tren Online Boutique). Tu so van la so gia tang do co
                # che du bao (projected - baseline), vi do la DAI LUONG mo
                # hinh sinh ra; hieu so nen sai khac mean/median phan lon triet
                # tieu.
                delta = float(proj) - float(base)
                span = ceiling - floor

                def _consumed(ceil_val):
                    hb = ceil_val - floor
                    return None if hb <= 0 else delta / hb

                c_mid = _consumed(ceiling)
                # tran LON hon -> bien rong hon -> tieu it hon, va nguoc lai
                c_lo, c_hi = _consumed(hi_c), _consumed(lo_c)
                if c_mid is None:
                    # P50 == P99: node gan nhu khong bien thien trong du lieu
                    # quan sat duoc -> khong co "bien van hanh" nao de do.
                    rec.update({'headroom_before': 0.0, 'consumed_pct': None,
                                'remaining_pct': 0.0, 'status': 'NO_OBSERVED_RANGE',
                                'reason': 'P50 == P99: khong quan sat duoc bien thien nao'})
                    node_out[metric_key] = rec
                    continue

                def _classify(c):
                    if c is None:
                        return 'NO_OBSERVED_RANGE'
                    if c > self.HEADROOM_CRITICAL_FRAC:
                        return 'BEYOND_ENVELOPE'
                    if c > self.HEADROOM_WARNING_FRAC:
                        return 'TIGHT'
                    return 'OK'

                st_lo, st_hi = _classify(c_lo), _classify(c_hi)
                # Khoang tin cay cua TRAN vat qua ranh gioi quyet dinh -> du
                # lieu khong du de phan dinh node nay. Khong can them hang so:
                # ranh gioi van la 2 nguong chinh sach da khai bao.
                st = _classify(c_mid) if st_lo == st_hi else 'UNDECIDED'
                rec.update({
                    'headroom_before': round(span, 4),
                    # consumed am = du bao GIAM so voi nen; giu nguyen dau vi do
                    # la thong tin that, nhung remaining_pct kep vao [0,100].
                    'consumed_pct': round(c_mid * 100, 1),
                    'consumed_ci_pct': [round(c_lo * 100, 1) if c_lo is not None else None,
                                        round(c_hi * 100, 1) if c_hi is not None else None],
                    'remaining_pct': round(min(max(1.0 - c_mid, 0.0), 1.0) * 100, 1),
                    'status': st,
                })
                node_out[metric_key] = rec
            if node_out:
                out[svc] = node_out
        return out

    def _ceiling_ci(self, col: str) -> Tuple[float, float]:
        """Khoang tin cay bootstrap cua tran P99 cho mot node.

        Ly do can: P99 duoc uoc luong tu mau huu han, va voi node co duoi
        nang (vd SockShop catalogue_cpu: P99=0.23 nhung max=49.0) no rat
        khong on dinh -- dung nhu bay co mau da lam hong uoc luong capacity_
        trong queueing_regressor.py. Bootstrap de bien su khong on dinh do
        thanh mot dai luong BAO CAO DUOC thay vi mot nguong tuy chon.
        """
        cache = getattr(self, '_ceiling_ci_cache', None)
        if cache is None:
            cache = self._ceiling_ci_cache = {}
        if col in cache:
            return cache[col]
        vals = (self.global_df_baseline[col].dropna().values
                if self.global_df_baseline is not None and col in self.global_df_baseline.columns
                else np.array([]))
        if len(vals) < 20:
            p99 = float(np.percentile(vals, 99)) if len(vals) else 0.0
            cache[col] = (p99, p99)
            return cache[col]
        rng = np.random.default_rng(42)
        n = min(len(vals), self.CEILING_BOOTSTRAP_N)
        draws = rng.choice(vals, size=(self.CEILING_BOOTSTRAP_B, n), replace=True)
        boots = np.percentile(draws, 99, axis=1)
        cache[col] = (float(np.percentile(boots, self.CEILING_CI_LOW)),
                      float(np.percentile(boots, self.CEILING_CI_HIGH)))
        return cache[col]

    @staticmethod
    def summarize_headroom(headroom: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Gop headroom thanh (status tong, danh sach dong canh bao doc duoc).

        Status tong lay theo node TE NHAT: co node BEYOND_ENVELOPE -> CRITICAL;
        co node TIGHT -> WARNING; con lai SAFE.

        UNDECIDED va NO_OBSERVED_RANGE co y KHONG nang muc canh bao: ca hai
        deu la "du lieu khong du de phan dinh", khac han voi "da do duoc va
        thay nguy hiem". De chung day phan quyet len CRITICAL la bien thieu
        thong tin thanh bao dong -- van liet ke de nguoi doc thay.
        """
        worst, lines = 'SAFE', []
        for svc, metrics in (headroom or {}).items():
            for metric, h in metrics.items():
                st = h.get('status')
                if st == 'BEYOND_ENVELOPE':
                    worst = 'CRITICAL'
                elif st == 'TIGHT' and worst != 'CRITICAL':
                    worst = 'WARNING'
                # UNDECIDED co y KHONG nang muc canh bao: khoang tin cay cua
                # tran vat qua ranh gioi nen du lieu khong du phan dinh -- de
                # no quyet dinh phan quyet la bao dong gia. Van liet ke kem
                # khoang de nguoi doc tu thay do rong.
                if st in ('BEYOND_ENVELOPE', 'TIGHT', 'UNDECIDED', 'NO_OBSERVED_RANGE'):
                    pct = h.get('consumed_pct')
                    ci = h.get('consumed_ci_pct')
                    note = f", khoang {ci[0]}-{ci[1]}%" if ci and None not in ci else ''
                    lines.append(f"{svc}.{metric}: tieu {pct}% bien du phong{note} "
                                  f"({h.get('baseline')} -> {h.get('projected')}, "
                                  f"tran P99={h.get('ceiling_p99')}) [{st}]")
        return worst, sorted(lines)

    # ----------------------------------------------------------
    def _apply_chain_correction(self, sim_result: Dict[str, Any], chain: set) -> Dict[str, Any]:
        """Sua hau qua da do duoc (papers/p1_du_phong/experiments/model_eval/evaluate_chain_awareness.py,
        994 diem tu SS-LIMITS/SS-LIMITS-CLEAN): simulate_intervention() lan Delta
        theo ty le workload LICH SU (rho_s), khong gioi han vao chain cua archetype
        -- du bao THAP hon thuc te o node TRONG chain (rui ro false-feasible) va
        THUA ra o node NGOAI chain.

        CHI sua truong 'cpu_predicted'/'cpu_change_pct' -- mem/latency/socket GIU
        NGUYEN vi correction chi duoc kiem chung cho CPU. Khong sua gi neu
        self._chain_correction rong (thieu file calibration) hoac `chain` rong.
        """
        if not self._chain_correction or not chain:
            return sim_result
        log_a, b = self._chain_correction['log_a'], self._chain_correction['b']
        out = {}
        for svc, entry in (sim_result or {}).items():
            e = dict(entry)
            base, pred = e.get('cpu_baseline'), e.get('cpu_predicted')
            if base is None or pred is None or pred == float('inf'):
                out[svc] = e
                continue
            if svc not in chain:
                # Ngoai chain: tinh nang nay khong lan tai toi day -- giu CPU = baseline.
                e['cpu_predicted'] = base
                e['cpu_change_pct'] = 0.0
                e['chain_corrected'] = True
            else:
                rho_s = self.rho.get(svc)
                if rho_s and rho_s > 0:
                    ratio = float(np.exp(log_a + b * np.log(1.0 / rho_s)))
                    delta_cpu = (pred - base) * ratio
                    e['cpu_predicted'] = base + delta_cpu
                    e['cpu_change_pct'] = round((delta_cpu / abs(base)) * 100.0, 1) if base else e.get('cpu_change_pct')
                    e['chain_corrected'] = True
            out[svc] = e
        return out

    # ----------------------------------------------------------
    def _status_at_L(self, L: float, inj_svc: str, delta_scaled_pct: float,
                      affected: set, base_wl: float) -> Dict[str, Any]:
        """Dung CHINH co che cua assess_capacity() (simulate_intervention +
        _apply_chain_correction + compute_headroom), tai MOT tai dinh L --
        KHONG goi LLM (khac assess_capacity(), bo qua buoc REASON de quet
        duoc nhieu diem). Cong thuc bac cau GIONG HET
        evaluate_verdict_level_heA.py (da kiem khop 10/10 phan quyet do
        truoc khi dung trong file do):
            delta_pct_eff(L) = (L*(1+delta_scaled_pct/100)/base_wl - 1) * 100
        Tra ve {'status_mid','status_lo','status_hi','headroom'} -- lo/hi suy
        TRUC TIEP tu consumed_ci_pct da co san trong compute_headroom() (CI
        bootstrap cua tran P99), khong tao co che bat dinh moi."""
        delta_pct_eff = (L * (1.0 + delta_scaled_pct / 100.0) / base_wl - 1.0) * 100.0
        sim = self.simulate_intervention(inj_svc, delta_pct_eff)
        sim = self._apply_chain_correction(sim, affected)
        headroom = self.compute_headroom(sim)

        def _classify_frac(c):
            if c is None:
                return None
            if c > self.HEADROOM_CRITICAL_FRAC:
                return 'BEYOND_ENVELOPE'
            if c > self.HEADROOM_WARNING_FRAC:
                return 'TIGHT'
            return 'OK'

        def _worst(bound: str) -> str:
            worst = 'SAFE'
            for metrics in (headroom or {}).values():
                for h in metrics.values():
                    if bound == 'mid':
                        st = h.get('status')
                        hit = st == 'BEYOND_ENVELOPE', st == 'TIGHT'
                    else:
                        ci = h.get('consumed_ci_pct')
                        if not ci or None in ci:
                            continue  # NO_OBSERVED_RANGE/not_forecastable/unavailable -- khong tinh
                        c = (ci[0] if bound == 'lo' else ci[1]) / 100.0
                        cls = _classify_frac(c)
                        hit = cls == 'BEYOND_ENVELOPE', cls == 'TIGHT'
                    if hit[0]:
                        worst = 'CRITICAL'
                    elif hit[1] and worst != 'CRITICAL':
                        worst = 'WARNING'
            return worst

        return {'status_mid': _worst('mid'), 'status_lo': _worst('lo'),
                'status_hi': _worst('hi'), 'delta_pct_eff': round(delta_pct_eff, 4)}

    def find_breakpoint(
        self,
        parsed_requirement: Dict[str, Any],
        impact_graph: Dict[str, Any] = None,
        L_floor: float = 1.0,
        L_cap: float = 2.0e5,
        read: str = 'lenient',
    ) -> Dict[str, Any]:
        """Quet tai dinh L (truoc khi co tinh nang) tim diem gay R* -- tai NHO
        NHAT ma trang thai tong KHONG con an toan.

        MOI, chua co o ban nao truoc: He A tu no (assess_capacity) chi tung
        bao trang thai TAI MOT diem co dinh (delta neo cua archetype, khong
        lien quan toi mot tai dinh L cu the nao) -- khong tung du bao duoc
        "gay o dau". Ham nay lap day dung khoang trong do, dung lai NGUYEN
        co che cua assess_capacity() (sim + chain correction + headroom),
        KHONG goi LLM trong luc quet (chi goi nhieu lan ham so, re).

        `read`: 'lenient' (mac dinh) -- chi CRITICAL tinh la gay (WARNING van
        coi la con dung duoc, chi la can luu y); 'strict' -- CRITICAL hoac
        WARNING deu tinh la gay. Hai cach doc nay cho ra R* khac nhau ro ret
        (da do o evaluate_verdict_level_heA.py) -- KHONG co cach "dung" duy
        nhat, goi ro rang trong ket qua tra ve de khong lan.

        GIA DINH DON DIEU (tai tang thi trang thai khong giam): cung co so
        voi _check_evidence_sanity/Phan A trong evaluate_verdict_level_heA.py
        -- moi co che CPU trong DAG la LinearRegression(positive=True).
        KHONG tu kiem lai gia dinh nay o day (da kiem 1 lan o script danh
        gia, tren luoi 288 diem, 0 vi pham -- xem bao_cao_giao_vien.tex).
        Neu gia dinh sai cho mot he/metric cu the, bisection co the tra ve
        sai -- day la dieu kien tien quyet CHUA duoc kiem lai moi lan goi.

        Tra ve {'R_star': {'mid','optimistic','conservative'}, 'read',
        'delta_scaled_pct', 'found': bool}. 'optimistic' suy tu dau CI LAC
        QUAN cua tran (it tieu tai nguyen hon) -- CHO RA SO LON HON ('mid'),
        vi he tro nen an toan lau hon. 'conservative' suy tu dau CI BI QUAN
        -- CHO RA SO NHO HON, nen dung lam can cu ke hoach dung luong (truong
        hop xau nhat con trong CI da do). KHONG dat ten 'lo'/'hi' (de lan voi
        huong cua R* thay vi huong cua CI) -- da tu bat loi nay khi viet, doi
        lai truoc khi dung. R_star[x] la None neu khong tim duoc trong
        [L_floor, L_cap] (an toan xuyen suot toan khoang, hoac da khong-an-
        toan ngay tu L_floor -- ca hai truong hop deu KHONG doan, bao ro
        bang None + ghi trong 'note')."""
        inj_svc = parsed_requirement.get('injection_service', self.default_injection)
        affected = set(parsed_requirement.get('affected_services', []))
        if impact_graph:
            for srv, deps in impact_graph.items():
                affected.update(deps.get('api_consumers_to_notify', []))
                affected.update(deps.get('downstream_services_to_check', []))
        delta_scaled_pct = float(parsed_requirement.get('injection_delta_pct', 20.0))

        gw_col = f'{inj_svc}_workload'
        if self.df_baseline is None or gw_col not in self.df_baseline.columns:
            return {'R_star': {'mid': None, 'lo': None, 'hi': None}, 'read': read,
                    'delta_scaled_pct': delta_scaled_pct, 'found': False,
                    'note': f"khong co cot '{gw_col}' trong du lieu baseline -- khong quet duoc"}
        base_wl = float(self.df_baseline[gw_col].mean())

        def _is_broken(status_dict, bound):
            st = status_dict[f'status_{bound}']
            return st == 'CRITICAL' if read == 'lenient' else st != 'SAFE'

        def _bisect(bound):
            lo_L, hi_L = L_floor, L_cap
            s_lo = self._status_at_L(lo_L, inj_svc, delta_scaled_pct, affected, base_wl)
            if _is_broken(s_lo, bound):
                return lo_L  # da gay ngay tu san -- khong bisect duoc, bao ro L_floor
            s_hi = self._status_at_L(hi_L, inj_svc, delta_scaled_pct, affected, base_wl)
            if not _is_broken(s_hi, bound):
                return None  # an toan xuyen suot toi L_cap -- khong tim thay trong khoang
            for _ in range(40):  # du de hoi tu voi L_cap=2e5, L_floor=1
                mid_L = (lo_L + hi_L) / 2.0
                s_mid = self._status_at_L(mid_L, inj_svc, delta_scaled_pct, affected, base_wl)
                if _is_broken(s_mid, bound):
                    hi_L = mid_L
                else:
                    lo_L = mid_L
            return round(hi_L, 2)

        # luu y: _bisect('lo')/_bisect('hi') dung dung ten khoa 'status_lo'/'status_hi'
        # cua _status_at_L() (huong CI tieu thu THAP/CAO) -- 'lo' (tieu it hon) cho ra
        # R* LON hon (optimistic); 'hi' (tieu nhieu hon) cho ra R* NHO hon (conservative).
        r_mid, r_optimistic, r_conservative = _bisect('mid'), _bisect('lo'), _bisect('hi')
        return {
            'R_star': {'mid': r_mid, 'optimistic': r_optimistic, 'conservative': r_conservative},
            'read': read, 'delta_scaled_pct': delta_scaled_pct,
            'found': r_mid is not None,
            'note': ('khong tim thay diem gay trong khoang quet' if r_mid is None
                     and r_conservative is None else None),
        }

    # ----------------------------------------------------------
    def assess_capacity(
        self,
        parsed_requirement: Dict[str, Any],
        impact_graph: Dict[str, Any] = None
    ) -> CapacityAssessment:
        """
        Thực thi chu trình ReAct hoàn chỉnh:
          1. ACT: Gọi SCM Tool (Multi-hop Accurate Path -- Global DAG).
          2. OBSERVE: Tổng hợp delta, nhận diện điểm nghẽn tiệm cận ngưỡng an toàn.
          3. REASON & CRITIQUE: LLM hoặc Domain Expert Logic phân tích nguyên nhân & phản biện rủi ro.
        """
        delta = float(parsed_requirement.get('injection_delta_pct', 20.0))
        inj_svc = parsed_requirement.get('injection_service', self.default_injection)
        affected = set(parsed_requirement.get('affected_services', []))

        # --- 1. ACT (Chạy SCM Tool) ---
        print(f"\n[CapacityAgent: Act] Chạy Accurate-Path SCM cho delta=+{delta}% tại '{inj_svc}'...")
        sim_result = self.simulate_intervention(inj_svc, delta)
        sim_result = self._apply_chain_correction(sim_result, affected)

        # --- 2. OBSERVE: quy doi sang BIEN DU PHONG ---
        # Truoc day buoc nay dung nguong tuyet doi (`cpu_pred > 70.0`) gia dinh
        # CPU la PHAN TRAM; trong RCAEval/Alibaba cot _cpu la core/byte nen
        # nguong do gan nhu khong bao gio kich hoat, va nhanh latency (`lat_chg
        # > 40`) thi dua tren dai luong da chung minh la khong du bao duoc.
        # Nay quyet dinh dua tren % bien du phong bi tieu, so voi tran P99 cua
        # CHINH tung node -- xem compute_headroom().
        headroom = self.compute_headroom(sim_result)
        status, saturated_services = self.summarize_headroom(headroom)

        # --- 3. REASON & CRITIQUE (Suy luận chuyên sâu + Devil's Advocate) ---
        print(f"[CapacityAgent: Reason] Phân tích điểm nghẽn & Tự phản biện rủi ro...")
        expert_assessment, risk_critique, recommendations = self._reason_and_critique(
            parsed_requirement,
            sim_result,
            saturated_services,
            status
        )

        return CapacityAssessment(
            status             = status,
            simulation_result  = sim_result,
            saturated_services = saturated_services,
            expert_assessment  = expert_assessment,
            risk_critique      = risk_critique,
            recommendations    = recommendations,
            confidence         = parsed_requirement.get('confidence', 'MEDIUM'),
            ood_confidence     = self._last_ood_confidence,
            ood_flagged_nodes  = self._last_ood_flagged,
            headroom           = headroom,
        )

    # ----------------------------------------------------------
    # INTERNAL: REASONING & DEVIL'S ADVOCATE LLM PROMPT
    # ----------------------------------------------------------
    def _reason_and_critique(
        self,
        parsed_req: Dict[str, Any],
        sim_result: Dict[str, Any],
        saturated_services: List[str],
        status: str
    ) -> Tuple[str, str, List[str]]:
        """
        Thực hiện bước Reasoning & Self-Critique thông qua LLM hoặc Heuristic.
        """
        if self.llm is None:
            expert_txt = (
                f"Phân tích tải: Hệ thống ở trạng thái {status}. "
                f"Đã kiểm tra {len(sim_result)} dịch vụ. "
                f"Điểm nghẽn tiềm ẩn: {', '.join(saturated_services) if saturated_services else 'Không có dịch vụ nào vượt ngưỡng'}. "
                "Cascade Attenuation diễn ra ổn định qua đồ thị vết gọi."
            )
            risk_critique = (
                "[Devil's Advocate Self-Critique]: Lưu ý mô hình tuyến tính có thể đánh giá thấp "
                "hiện tượng nghẽn hàng đợi (Queueing delay) khi tải vượt ngưỡng 80%. "
                "Cần kiểm tra thêm dung lượng Connection Pool của Database."
            )
            recs = [
                "Cân nhắc scale pod của các core services trước khi bật cờ tính năng (Feature Flag).",
                "Theo dõi chặt chẽ Socket connection count trong 30 phút đầu rollout."
            ]
            return expert_txt, risk_critique, recs

        prompt = f"""Bạn là Capacity & Performance Engineer chuyên gia vi dịch vụ SockShop.
Dưới đây là kết quả đo lường định lượng từ bộ công cụ nhân quả SCM (Global Causal DAG):

[YÊU CẦU TÍNH NĂNG MỚI]
- Loại: {parsed_req.get('request_type')}
- Gateway inject: {parsed_req.get('injection_service')} (+{parsed_req.get('injection_delta_pct')}%)
- Core Services (sửa code): {parsed_req.get('core_services')}
- Affected Services (blast radius): {parsed_req.get('affected_services')}

[DỮ LIỆU ĐO LƯỜNG SCM (Tool Output)]
1. Multi-hop Cascade (Global DAG):
{json.dumps(sim_result, indent=2)}

2. Dịch vụ cảnh báo bão hòa sơ bộ: {saturated_services if saturated_services else 'None'}

[NHIỆM VỤ REASONING (Suy Luận & Phản Biện Chuyên Sâu)]:
1. "expert_assessment": Giải thích ngắn gọn (3-4 câu) tại sao các dịch vụ bị tăng tải, và tại sao có hiệu ứng suy giảm (cascade attenuation) từ gateway đến các tầng sâu (như orders/payment).
2. "risk_critique" (DEVIL'S ADVOCATE): Hãy tự phản biện: "Có rủi ro ngoại sinh, giả định mô hình, hay nguy cơ tiềm ẩn nào (như bão hòa hàng đợi, nghẽn connection pool, memory leak, hoặc rủi ro ngoại suy OOD) mà số liệu trên có thể bỏ sót không?"
3. "recommendations": Đưa ra 2-3 giải pháp kỹ thuật cụ thể (ví dụ: bổ sung Redis cache, tăng replica, bật circuit breaker).

Trả về DUY NHẤT định dạng JSON sau:
{{
  "expert_assessment": "...",
  "risk_critique": "...",
  "recommendations": ["khuyến nghị 1", "khuyến nghị 2"]
}}"""

        try:
            from langchain_core.messages import HumanMessage
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            exp = data.get("expert_assessment", "Đánh giá hiệu năng hoàn tất.")
            crit = data.get("risk_critique", "[Devil's Advocate]: Cần kiểm tra kỹ năng lực DB.")
            recs = data.get("recommendations", ["Áp dụng caching và cân bằng tải."])
            return exp, crit, recs

        except Exception as e:
            print(f"  [CapacityAgent] Lỗi LLM Reasoning: {e}")
            return (
                f"Đánh giá bán tự động: Phát hiện {len(saturated_services)} nút cảnh báo.",
                "[Devil's Advocate]: Nguy cơ tiềm ẩn về đột biến độ trễ hàng đợi khi tải thực tế biến thiên.",
                ["Khuyến nghị kiểm tra log và cấu hình auto-scaling."]
            )
