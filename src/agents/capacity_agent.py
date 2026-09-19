# -*- coding: utf-8 -*-
"""
CapacityAgent — Unified SCM Performance & Capacity Reasoner
===========================================================
Hợp nhất PerformanceAgent (Fast Path Bivariate) và SimulationAgent (Accurate Global DAG)
thành một Agent duy nhất tuân thủ mô hình ReAct (Yao et al., 2022):

  1. ACT (Hành động qua Tool):
     - Chạy Dual-Path SCM Simulation:
       + Fast Path: 21 mô hình Bivariate độc lập (đánh giá trực tiếp per-service).
       + Accurate Path: Siêu đồ thị 28-node Causal DAG (mô phỏng lan truyền suy giảm cascade).

  2. OBSERVE (Quan sát định lượng):
     - Tổng hợp độ lệch tài nguyên (CPU, Memory, Latency, Socket).
     - Quét ngưỡng bão hòa (>70% baseline hoặc bùng nổ độ trễ phi tuyến).

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
    load_normal_data,
    load_multi_service_data,
    split_quantile,
    SERVICES,
    TRAINTICKET_SERVICES,
    METRICS
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

N_PROJ = 500


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def _load_normal_data(service: str, metric_col: str, data_dir: str) -> pd.DataFrame:
    """Load normal-period [Workload, Target] từ tất cả các run."""
    dfs = []
    wlc = f'{service}_workload'
    tgc = f'{service}_{metric_col}'

    target_dir = os.path.join(data_dir, 'RE2-SS') if os.path.isdir(os.path.join(data_dir, 'RE2-SS')) else data_dir

    for scenario in os.listdir(target_dir):
        sp = os.path.join(target_dir, scenario)
        if not os.path.isdir(sp):
            continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp):
                continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)):
                continue
            try:
                with open(ip) as f:
                    it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None or wlc not in cols or tgc not in cols:
                    continue
                df_raw = pd.read_csv(mp, usecols=[tc, wlc, tgc])
                df = df_raw[[tc, wlc, tgc]].copy()
                df.columns = [tc, 'Workload', 'Target']
                dfs.append(df[df[tc] < it].drop(columns=[tc]).dropna())
            except Exception:
                continue

    return pd.concat(dfs, ignore_index=True) if dfs else None


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
    fast_metrics: Dict[str, Any]        # Fast Path: Bivariate predictions
    simulation_result: Dict[str, Any]   # Accurate Path: Global DAG multi-hop cascade
    saturated_services: List[str]       # Danh sách service vượt ngưỡng an toàn (>70%)
    expert_assessment: str              # Phân tích nguyên nhân điểm nghẽn từ LLM / Rule
    risk_critique: str                  # Tự phản biện Devil's Advocate (rủi ro tiềm ẩn)
    recommendations: List[str]          # Các khuyến nghị kỹ thuật cụ thể
    confidence: str = "MEDIUM"          # "HIGH" | "MEDIUM" | "LOW"
    ood_confidence: str = "high"        # G7: "high"|"medium"|"low"|"very_low" — mức tin cậy
                                         # ngoại suy của *độ lớn* workload dự phóng, độc lập
                                         # với `confidence` (vốn chỉ đo taxonomy-membership).
    ood_flagged_nodes: List[str] = field(default_factory=list)  # node nào rơi ngoài P95 train


# ============================================================
# 3. CAPACITY AGENT (UNIFIED PERFORMANCE & SIMULATION AGENT)
# ============================================================
class CapacityAgent:
    """
    Agent Đánh Giá Năng Lực & Hiệu Năng Hệ Thống (ReAct Agent).
    
    Tích hợp:
      - Tool 1 (Fast Path): 21 mô hình Bivariate SCM độc lập.
      - Tool 2 (Accurate Path): Siêu đồ thị 28-node Causal DAG phản ánh cascade attenuation.
      - ReAct Loop: Act (chạy 2 chế độ SCM) -> Observe (tổng hợp delta) -> Reason & Critique (LLM/Expert logic).
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

        # State của Tool Fast Path (Bivariate)
        self.trained_models: Dict[Tuple[str, str], Any] = {}
        self.bivariate_models = self.trained_models
        self.accuracy_df: pd.DataFrame = None

        # State của Tool Accurate Path (Global DAG)
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

        self._is_trained = False

        print(f"[CapacityAgent] Khởi tạo ({self.system_type}). Data dir: {self.data_dir}")
        if auto_train:
            self.train()

    # ----------------------------------------------------------
    # HUẤN LUYỆN SCM TOOLS (CẢ 2 CHẾ ĐỘ)
    # ----------------------------------------------------------
    def train(self):
        """Huấn luyện đồng thời cả Fast Path (Bivariate) và Accurate Path (Global DAG)."""
        self.train_fast_path()
        self.train_accurate_path()
        self._is_trained = bool(self.trained_models and self.global_dag_model)
        self._key_nodes_cache = None  # DAG moi -> bo cache select_key_nodes() cu
        self._stability_cache = None  # DAG moi -> bo cache evaluate_node_stability() cu

    def train_fast_path(self):
        """Huấn luyện mô hình SCM Bivariate (N services x 3 metrics)."""
        print("\n" + "=" * 70)
        print(f"  [CapacityAgent: Tool 1] HUẤN LUYỆN BIVARIATE SCM ({self.system_type.upper()})")
        print("  Phương pháp: Train(LOW workload 67%) -> Test(HIGH workload 33%)")
        print("=" * 70)

        if not os.path.isdir(self.data_dir):
            print(f"[WARNING] Thư mục dữ liệu không tồn tại: {self.data_dir}")
            return

        try:
            df_multi = load_multi_service_data(self.data_dir, system_type=self.system_type, services=self.services)
        except Exception:
            df_multi = None

        all_rows = []

        for metric_name, metric_col, unit, scale in METRICS:
            print(f"\n  [{metric_name} | don vi: {unit}]")
            for svc in self.services:
                wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
                if df_multi is not None and wlc in df_multi.columns and tgc in df_multi.columns:
                    df = df_multi[[wlc, tgc]].dropna()
                    df.columns = ['Workload', 'Target']
                else:
                    df = _load_normal_data(svc, metric_col, self.data_dir)

                if df is None or len(df) < 200:
                    print(f"    {svc:<14}: khong du du lieu (can >=200 mau)")
                    continue

                if len(df) > 2000:
                    df = df.sample(2000, random_state=42)

                df = df.sort_values('Workload').reset_index(drop=True)
                split = int(len(df) * 0.67)
                df_train = df.iloc[:split]
                df_test  = df.iloc[split:].copy()

                g = nx.DiGraph()
                g.add_edge('Workload', 'Target')
                model = gcm.InvertibleStructuralCausalModel(g)

                # Ép dùng LinearRegression(positive=True) thay vì gcm.auto
                # tự chọn — đồng bộ với Global DAG (train_accurate_path).
                # Bằng chứng: experiments/nonlinear_mechanism_trial.py cho
                # thấy linear_pos thắng auto_gcm rõ rệt ở protocol quantile
                # (MAPE 12.9% vs 20.8%, win 10/21 vs 4/21 cặp), đúng chế
                # độ agent thực sự gọi model (new_wl = base*(1+delta%) —
                # luôn ngoại suy nhẹ so với baseline huấn luyện).
                model.set_causal_mechanism(
                    'Workload', EmpiricalDistribution())
                model.set_causal_mechanism(
                    'Target',
                    AdditiveNoiseModel(SklearnRegressionModel(
                        LinearRegression(positive=True))))
                gcm.fit(model, df_train)

                df_test['bkt'] = pd.qcut(
                    df_test['Workload'],
                    q=min(8, df_test['Workload'].nunique()),
                    duplicates='drop'
                )
                bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

                y_true, y_pred = [], []
                for _, row in bkts.iterrows():
                    wl_val = row['Workload']
                    dp = gcm.interventional_samples(
                        model,
                        interventions={'Workload': lambda x, w=wl_val: w},
                        num_samples_to_draw=N_PROJ
                    )
                    y_true.append(row['Target'])
                    y_pred.append(dp['Target'].mean())

                yt, yp = np.array(y_true), np.array(y_pred)
                mape_v = _mape(yt, yp)
                mae_v  = mean_absolute_error(yt, yp) * scale
                rmse_v = np.sqrt(mean_squared_error(yt, yp)) * scale
                r2_v   = r2_score(yt, yp)

                thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
                yt_bin = (yt >= thresh).astype(int)
                yp_bin = (yp >= thresh).astype(int)
                f1_v   = f1_score(yt_bin, yp_bin, average='binary', zero_division=1)

                tag = 'EXCELLENT' if mape_v < 10 else ('FAIR' if mape_v < 25 else 'POOR')
                print(f"    {svc:<14}: MAPE={mape_v:>5.1f}%  "
                      f"RMSE={rmse_v:>8.4f}  F1={f1_v:>5.3f}  R2={r2_v:>5.3f}  [{tag}]")

                self.trained_models[(svc, metric_name)] = {
                    'model':        model,
                    'baseline_wl':  df_train['Workload'].mean(),
                    'baseline_val': df_train['Target'].mean() * scale,
                    'wl_train_max': df_train['Workload'].max(),
                    'wl_test_max':  df_test['Workload'].max(),
                    'unit':         unit,
                    'scale':        scale,
                    'mape':         mape_v,
                    'f1':           f1_v,
                    'r2':           r2_v,
                    'rmse':         rmse_v,
                    'thresh':       thresh,
                }

                all_rows.append({
                    'service':      svc,
                    'metric':       metric_name,
                    'unit':         unit,
                    'mape_pct':     round(mape_v, 2),
                    'rmse':         round(rmse_v, 4),
                    'mae':          round(mae_v, 4),
                    'r2':           round(r2_v, 3),
                    'f1_score':     round(f1_v, 3),
                    'tag':          tag,
                    'baseline_val': round(df_train['Target'].mean() * scale, 4),
                    'n_train':      len(df_train),
                    'n_test':       len(df_test),
                })

        self.accuracy_df = pd.DataFrame(all_rows)
        print(f"\n[OK] Đã huấn luyện {len(self.trained_models)} mô hình Bivariate SCM.")

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
        g.add_edges_from(edge_build['all_edges'])
        if edge_build.get('from_cache'):
            print(f"  [Edges] dung cache: {len(edge_build['learned_edges'])} canh da hoc.")
        for label, rep in edge_build['reports'].items():
            print(f"  [{label}] {rep['n_candidates']} candidate -> {rep['n_selected']} qua "
                  f"held-out/knee-point -> {rep['n_final']} qua OOD-safety.")

        valid_nodes = [n for n in g.nodes() if n in df_data.columns]
        g_sub = g.subgraph(valid_nodes).copy()
        df_sub = df_data[valid_nodes].dropna()

        # Baseline dữ liệu bình thường
        self.global_df_baseline = df_sub
        self.df_baseline      = self.global_df_baseline
        self.dag_graph        = g_sub
        self.dag              = self.dag_graph

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
    # BACKWARD COMPATIBILITY INTERFACES (Cho các script cũ)
    # ----------------------------------------------------------
    def train_scm_models(self):
        """Tương thích PerformanceAgent.train_scm_models()."""
        self.train_fast_path()

    def get_metrics_for_service(
        self,
        service_name: str,
        workload_delta_pct: float = 20.0,
        scenario: str = None
    ) -> Dict[str, Any]:
        """Tương thích PerformanceAgent.get_metrics_for_service()."""
        if not self.trained_models:
            return {}

        result = {}
        for metric_name, metric_col, unit, scale in METRICS:
            key = (service_name, metric_name)
            if key not in self.trained_models:
                continue

            m_info  = self.trained_models[key]
            model   = m_info['model']
            base_wl = m_info['baseline_wl']
            base_v  = m_info['baseline_val']
            new_wl  = base_wl * (1 + workload_delta_pct / 100)

            dp = gcm.interventional_samples(
                model,
                interventions={'Workload': lambda x, w=new_wl: w},
                num_samples_to_draw=N_PROJ
            )
            pred_v = dp['Target'].mean() * scale
            chg    = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0

            result[f'{metric_name}_baseline_{unit}']  = round(base_v, 4)
            result[f'{metric_name}_predicted_{unit}'] = round(pred_v, 4)
            result[f'{metric_name}_change_pct']       = round(chg, 2)
            # Do tin cay cua CHINH node nay trong Global DAG (_node_confidence(),
            # dua tren evaluate_node_stability() -- KHONG loc qua top-impact
            # toan cuc, xem docstring _get_key_nodes()) -- Fast Path (Bivariate)
            # va Accurate Path
            # (DAG) la 2 mo hinh khac nhau tren CUNG mot node ten, nhung day
            # la tin hieu duy nhat co san ve do on dinh cua node do trong toan
            # do thi (Fast Path khong tu danh gia rieng cho tung service; xem
            # accuracy_df/get_accuracy_report() cho do chinh xac RIENG cua
            # CHINH mo hinh Bivariate nay neu can phan biet 2 nguon).
            result[f'{metric_name}_confidence'] = self._node_confidence(f'{service_name}_{metric_col}')

        return result

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
        """Tương thích SimulationAgent.simulate_intervention()."""
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

        samples = gcm.interventional_samples(
            self.global_dag_model,
            interventions={injection_col: lambda x, w=target_wl: w},
            num_samples_to_draw=n_samples
        )

        # G7: classify extrapolation risk on the samples we already have --
        # stashed on self, read by assess_capacity() right after this call.
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
                if col in self.df_baseline.columns and col in samples.columns:
                    base = float(self.df_baseline[col].mean())
                    pred = float(samples[col].mean())
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

        if injection_service is None:
            injection_service = self.default_injection
        inj_col = f"{injection_service}_workload"
        if inj_col not in self.global_df_baseline.columns:
            return {}

        base_wl = float(self.global_df_baseline[inj_col].mean())
        w_lo = base_wl * (1.0 + delta_min_pct / 100.0)
        w_hi = base_wl * (1.0 + delta_max_pct / 100.0)

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

    def get_accuracy_report(self) -> pd.DataFrame:
        if self.accuracy_df is None or self.accuracy_df.empty:
            return pd.DataFrame()
        return self.accuracy_df.copy()

    def get_accuracy_summary(self) -> Dict[str, Any]:
        if self.accuracy_df is None or self.accuracy_df.empty:
            return {}

        summary = {}
        for metric_name, _, _, _ in METRICS:
            sub = self.accuracy_df[self.accuracy_df['metric'] == metric_name]
            if sub.empty:
                continue
            n_good = int((sub['mape_pct'] < 10).sum())
            summary[metric_name] = {
                'avg_mape_pct':       round(float(sub['mape_pct'].mean()), 2),
                'avg_f1':             round(float(sub['f1_score'].mean()), 3),
                'avg_r2':             round(float(sub['r2'].mean()), 3),
                'excellent_services': n_good,
                'total_services':     len(sub),
            }
        return summary

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
    def assess_capacity(
        self,
        parsed_requirement: Dict[str, Any],
        impact_graph: Dict[str, Any] = None
    ) -> CapacityAssessment:
        """
        Thực thi chu trình ReAct hoàn chỉnh:
          1. ACT: Gọi 2 chế độ SCM Tool (Bivariate Fast Path + Multi-hop Accurate Path).
          2. OBSERVE: Tổng hợp delta, nhận diện điểm nghẽn tiệm cận ngưỡng an toàn.
          3. REASON & CRITIQUE: LLM hoặc Domain Expert Logic phân tích nguyên nhân & phản biện rủi ro.
        """
        delta = float(parsed_requirement.get('injection_delta_pct', 20.0))
        inj_svc = parsed_requirement.get('injection_service', self.default_injection)
        affected = set(parsed_requirement.get('affected_services', []))
        core_svcs = set(parsed_requirement.get('core_services', []))

        services_to_check = affected | core_svcs
        if impact_graph:
            for srv, deps in impact_graph.items():
                services_to_check.update(deps.get('api_consumers_to_notify', []))
                services_to_check.update(deps.get('downstream_services_to_check', []))

        # --- 1. ACT (Chạy SCM Tools) ---
        print(f"\n[CapacityAgent: Act] Chạy Dual-Path SCM cho delta=+{delta}% tại '{inj_svc}'...")
        fast_metrics = {}
        for srv in services_to_check:
            m = self.get_metrics_for_service(srv, workload_delta_pct=delta)
            if m:
                fast_metrics[srv] = m

        sim_result = self.simulate_intervention(inj_svc, delta)

        # --- 2. OBSERVE (Quan sát định lượng & Sàng lọc ngưỡng) ---
        saturated_services = []
        for srv, metrics in fast_metrics.items():
            cpu_pred = metrics.get('CPU_predicted_%', metrics.get('cpu_predicted_%', 0.0))
            cpu_chg  = metrics.get('CPU_change_pct', metrics.get('cpu_change_pct', 0.0))
            if cpu_chg > 25.0 or cpu_pred > 70.0:
                saturated_services.append(f"{srv} (CPU +{cpu_chg}%)")

        for srv, metrics in sim_result.items():
            lat_chg = metrics.get('latency_ms_change_pct', 0.0)
            if lat_chg > 40.0:
                saturated_services.append(f"{srv} (Latency +{lat_chg}%)")

        saturated_services = sorted(list(set(saturated_services)))

        # Xác định trạng thái sơ bộ
        if len(saturated_services) >= 2:
            status = "CRITICAL"
        elif len(saturated_services) == 1:
            status = "WARNING"
        else:
            status = "SAFE"

        # --- 3. REASON & CRITIQUE (Suy luận chuyên sâu + Devil's Advocate) ---
        print(f"[CapacityAgent: Reason] Phân tích điểm nghẽn & Tự phản biện rủi ro...")
        expert_assessment, risk_critique, recommendations = self._reason_and_critique(
            parsed_requirement,
            fast_metrics,
            sim_result,
            saturated_services,
            status
        )

        return CapacityAssessment(
            status             = status,
            fast_metrics       = fast_metrics,
            simulation_result  = sim_result,
            saturated_services = saturated_services,
            expert_assessment  = expert_assessment,
            risk_critique      = risk_critique,
            recommendations    = recommendations,
            confidence         = parsed_requirement.get('confidence', 'MEDIUM'),
            ood_confidence     = self._last_ood_confidence,
            ood_flagged_nodes  = self._last_ood_flagged,
        )

    # ----------------------------------------------------------
    # INTERNAL: REASONING & DEVIL'S ADVOCATE LLM PROMPT
    # ----------------------------------------------------------
    def _reason_and_critique(
        self,
        parsed_req: Dict[str, Any],
        fast_metrics: Dict[str, Any],
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
                f"Đã kiểm tra {len(fast_metrics)} dịch vụ. "
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
Dưới đây là kết quả đo lường định lượng từ 2 bộ công cụ nhân quả SCM (Dual-Path):

[YÊU CẦU TÍNH NĂNG MỚI]
- Loại: {parsed_req.get('request_type')}
- Gateway inject: {parsed_req.get('injection_service')} (+{parsed_req.get('injection_delta_pct')}%)
- Core Services (sửa code): {parsed_req.get('core_services')}
- Affected Services (blast radius): {parsed_req.get('affected_services')}

[DỮ LIỆU ĐO LƯỜNG SCM (Tool Outputs)]
1. Fast Path (Bivariate predictions):
{json.dumps(fast_metrics, indent=2)}

2. Accurate Path (Multi-hop Cascade Global DAG):
{json.dumps(sim_result, indent=2)}

3. Dịch vụ cảnh báo bão hòa sơ bộ: {saturated_services if saturated_services else 'None'}

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
