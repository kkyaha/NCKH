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
class QueueingLatencyRegressor(BaseEstimator, RegressorMixin):
    """
    Mô hình hàng đợi phi tuyến (M/M/1 - Kleinrock approximation):
    Khi Workload -> Capacity, Latency bùng nổ theo hàm tiệm cận: W / (C - W).
    """
    def __init__(self):
        self.model_ = LinearRegression(fit_intercept=True)
        self.capacity_ = None

    def fit(self, X, y):
        X = np.array(X)
        self.capacity_ = np.max(X, axis=0) * 1.5
        self.capacity_[self.capacity_ == 0] = 1.0
        X_queue = X / (self.capacity_ - X + 1e-6)
        X_transformed = np.hstack([X, X_queue])
        self.model_.fit(X_transformed, y)
        return self

    def predict(self, X):
        X = np.array(X)
        X_capped = np.minimum(X, self.capacity_ * 0.99)
        X_queue  = X_capped / (self.capacity_ - X_capped + 1e-6)
        return self.model_.predict(np.hstack([X_capped, X_queue]))


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
            self.default_injection = 'ts-preserve-service'
        else:
            if data_dir is None:
                data_dir = os.path.join(BASE_DIR, 'data', 'raw')
            if graph_path is None:
                graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
            self.services = services or SERVICES
            self.default_injection = 'front-end'

        self.llm        = llm
        self.data_dir   = data_dir
        self.graph_path = graph_path

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
            df_multi = load_multi_service_data(self.data_dir, system_type=self.system_type)
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

        df_data = load_multi_service_data(self.data_dir, system_type=self.system_type)
        if df_data is None or df_data.empty:
            print("[WARNING] Không load được dữ liệu đa dịch vụ.")
            return

        if not os.path.exists(self.graph_path):
            print(f"[WARNING] Không tìm thấy file graph: {self.graph_path}")
            return

        with open(self.graph_path, 'r', encoding='utf-8') as f:
            graph_json = json.load(f)

        g = nx.DiGraph()
        # Tier 1: Workload -> Workload theo call chain thực tế
        for edge in graph_json.get('edges', []):
            src, tgt = edge['source'], edge['target']
            if src in self.services and tgt in self.services:
                g.add_edge(f"{src}_workload", f"{tgt}_workload")

        # Tier 2: Workload -> Metrics nội tại
        for s in self.services:
            for metric_col in [f'{s}_cpu', f'{s}_mem', f'{s}_latency-50']:
                if f'{s}_workload' in df_data.columns and metric_col in df_data.columns:
                    g.add_edge(f"{s}_workload", metric_col)

        # Tier 2.5: canh "backpressure" tu CPU cua caller sang CPU cua callee —
        # danh sach RIENG cho tung he thong, moi canh da duoc kiem dinh bang du
        # lieu that (KHONG dong loat/mo rong tuy y cho toan bo node) qua 3 buoc:
        #   1) experiments/{call_chain_neighbor,tt_call_chain_neighbor}_diagnostic.py
        #      — R2 tang khi them CPU caller lam parent thu 2 (nguong gain>0.03
        #      cho Train Ticket; Sock Shop chon thu cong 3 canh manh nhat).
        #   2) experiments/replace_vs_add_edge_test.py — xac nhan phai THEM (giu
        #      ca workload rieng LAN caller_cpu), khong duoc THAY: thay se lam
        #      mat kha nang phan ung voi mot can thiep do() rieng le tai chinh
        #      callee (vd do(shipping_workload=+30%) voi orders_cpu giu nguyen ->
        #      mo hinh REPLACE du bao +0.00%, sai ro rang).
        #   3) experiments/{backpressure,tt_backpressure}_edge_accuracy_test.py —
        #      xac nhan tren du lieu HELD-OUT (protocol OOD Gold Standard giong
        #      RQ1, khong phai R2 in-sample): Sock Shop ca 3/3 canh cai thien
        #      MAPE/R2/F1; Train Ticket 49/50 canh giam MAPE, 47/50 tang R2/F1.
        #   4) experiments/{backpressure,tt_backpressure}_edge_ood_safety_test.py —
        #      quet delta +5%..+300%, dem so ca sign-inversion: Sock Shop giam
        #      hoi toan (30->24, sua duoc 2 loi co san); Train Ticket giam rong
        #      (35->26) nhung TANG nhe o dung +300% (8->11, cac node bi anh huong
        #      hau het KHONG phai 1-hop tu canh moi — nhieu kha nang la do do
        #      mong manh von co cua rang buoc tuyen tinh o cuc bien, khong rieng
        #      do canh nay) — nam trong pham vi +150%/+300% ma paper da tu gioi
        #      han la "minh hoa dinh huong, khong phai du bao da kiem chung"
        #      (Section "Extrapolation-Sign Failure Mode"), khong che giau caveat
        #      nay.
        if self.system_type == 'sockshop':
            BACKPRESSURE_EDGES = [
                ('orders_cpu', 'shipping_cpu'),
                ('orders_cpu', 'carts_cpu'),
                ('front-end_cpu', 'user_cpu'),
            ]
        elif self.system_type == 'trainticket':
            BACKPRESSURE_EDGES = [
                ('ts-ticketinfo-service_cpu', 'ts-basic-service_cpu'),
                ('ts-travel2-service_cpu', 'ts-train-service_cpu'),
                ('ts-travel2-service_cpu', 'ts-seat-service_cpu'),
                ('ts-seat-service_cpu', 'ts-config-service_cpu'),
                ('ts-basic-service_cpu', 'ts-train-service_cpu'),
                ('ts-travel2-service_cpu', 'ts-ticketinfo-service_cpu'),
                ('ts-travel2-service_cpu', 'ts-order-other-service_cpu'),
                ('ts-basic-service_cpu', 'ts-price-service_cpu'),
                ('ts-travel-service_cpu', 'ts-seat-service_cpu'),
                ('ts-seat-service_cpu', 'ts-order-other-service_cpu'),
                ('ts-basic-service_cpu', 'ts-route-service_cpu'),
                ('ts-travel-service_cpu', 'ts-train-service_cpu'),
                ('ts-travel-service_cpu', 'ts-ticketinfo-service_cpu'),
                ('ts-travel2-service_cpu', 'ts-route-service_cpu'),
                ('ts-food-service_cpu', 'ts-travel-service_cpu'),
                ('ts-travel-service_cpu', 'ts-route-service_cpu'),
                ('ts-food-service_cpu', 'ts-food-map-service_cpu'),
                ('ts-seat-service_cpu', 'ts-order-service_cpu'),
                ('ts-travel-service_cpu', 'ts-order-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-security-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-seat-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-contacts-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-ticketinfo-service_cpu'),
                ('ts-admin-basic-info-service_cpu', 'ts-price-service_cpu'),
                ('ts-admin-basic-info-service_cpu', 'ts-config-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-user-service_cpu'),
                ('ts-basic-service_cpu', 'ts-station-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-travel-service_cpu'),
                ('ts-order-other-service_cpu', 'ts-station-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-security-service_cpu'),
                ('ts-consign-service_cpu', 'ts-consign-price-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-user-service_cpu'),
                ('ts-security-service_cpu', 'ts-order-other-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-assurance-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-food-service_cpu'),
                ('ts-admin-travel-service_cpu', 'ts-travel2-service_cpu'),
                ('ts-admin-travel-service_cpu', 'ts-travel-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-travel2-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-seat-service_cpu'),
                ('ts-security-service_cpu', 'ts-order-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-assurance-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-contacts-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-ticketinfo-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-order-other-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-order-service_cpu'),
                ('ts-preserve-other-service_cpu', 'ts-food-service_cpu'),
                ('ts-food-service_cpu', 'ts-station-service_cpu'),
                ('ts-order-service_cpu', 'ts-station-service_cpu'),
                ('ts-inside-payment-service_cpu', 'ts-payment-service_cpu'),
                ('ts-preserve-service_cpu', 'ts-station-service_cpu'),
            ]
        else:
            BACKPRESSURE_EDGES = []
        for caller_cpu_node, callee_cpu_node in BACKPRESSURE_EDGES:
            if caller_cpu_node in df_data.columns and callee_cpu_node in df_data.columns:
                g.add_edge(caller_cpu_node, callee_cpu_node)

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

        # Gán cơ chế chuyên biệt: CPU/Mem + Tier-1 Workload->Workload (Linear,
        # rang buoc he so KHONG AM), Latency (Queueing phi tuyến).
        #
        # DONG BO VOI evaluation_suite.build_and_train_global_dag() (Sock Shop):
        # ban dau o day chi co _cpu/_mem dung LinearRegression() THUONG (khong
        # positive=True), va canh Tier-1 (Workload->Workload) khong duoc gan
        # rang buoc gi ca — tuc la CHINH XAC bug sign-inversion ma Section
        # "Extrapolation-Sign Failure Mode" cua paper mo ta da "fix" (nhung
        # fix do truoc day chi nam o evaluation_suite.py, mot script rieng
        # cho RQ4, KHONG nam trong CapacityAgent — class nay moi la code that
        # duoc orchestrator.py dung cho ca Sock Shop LAN Train Ticket). Ap
        # dung dung mot rang buoc cho ca 2 he thong o day de ket qua giua
        # Sock Shop va Train Ticket (vd RQ6 Part A.3) khong con lech nhau vi
        # mot confound ve quy trinh fit, chi con lech vi ban chat du lieu.
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

        gcm.fit(model, df_fit)
        self.global_dag_model = model
        self.global_model     = self.global_dag_model
        print(f"[OK] Đã khớp Global DAG {len(valid_nodes)} nodes ({len(g_sub.edges())} edges).")

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
        for metric_name, _, unit, scale in METRICS:
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
                if col in self.df_baseline.columns and col in samples.columns:
                    base = float(self.df_baseline[col].mean())
                    pred = float(samples[col].mean())
                    chg  = (pred - base) / abs(base) * 100.0 if base != 0 else 0.0
                    entry[metric_key] = round(chg, 1)
                    entry[f"{metric_key.replace('_change_pct', '')}_predicted"] = round(pred, 2)
                    entry[f"{metric_key.replace('_change_pct', '')}_baseline"]  = round(base, 2)
                else:
                    entry[metric_key] = None
            results[svc] = entry

        return results

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
