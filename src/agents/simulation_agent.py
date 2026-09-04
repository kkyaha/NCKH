# -*- coding: utf-8 -*-
"""
SimulationAgent — Multi-hop Global DAG Intervention Simulator
=============================================================
Vai tro trong dual-path architecture (Plan v4):
  ACCURATE PATH: do(injection_service_workload = baseline x (1 + delta/100))
                 -> SCM Global DAG 28-node tu lan truyen qua call chain
                 -> Phan anh cascade attenuation qua tung hop

Khac voi PerformanceAgent (Fast Path / Bivariate):
  - Bivariate: ap cung delta doc lap vao tung service, KHONG co cascade
  - SimulationAgent: 1 diem can thiep duy nhat, SCM tu lan truyen
                     (payment nhan it hon front-end vi nhieu hop)

Nguon: wrap build_and_train_global_dag() tu evaluation_suite.py
"""

import os
import sys
import warnings
import numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

# Path setup
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_AGENT_DIR)
BASE_DIR   = os.path.dirname(_SRC_DIR)

sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))

import networkx as nx
import pandas as pd
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

from data_processor import load_multi_service_data, SERVICES

N_SAMPLES = 500


# ============================================================
# QUEUEING LATENCY REGRESSOR (copy tu evaluation_suite.py)
# ============================================================
class QueueingLatencyRegressor(BaseEstimator, RegressorMixin):
    """
    Mo hinh hang doi M/M/1 cho latency: tang phi tuyen gan diem bao hoa.
    Domain-aware: tot hon LinearRegression khi ngoai suy vung tai cao.
    """
    def __init__(self):
        self.model_    = LinearRegression(fit_intercept=True)
        self.capacity_ = None

    def fit(self, X, y):
        X = np.array(X)
        self.capacity_ = np.max(X, axis=0) * 1.5
        self.capacity_[self.capacity_ == 0] = 1.0
        X_queue = X / (self.capacity_ - X + 1e-6)
        self.model_.fit(np.hstack([X, X_queue]), y)
        return self

    def predict(self, X):
        X       = np.array(X)
        X_capped = np.minimum(X, self.capacity_ * 0.99)
        X_queue  = X_capped / (self.capacity_ - X_capped + 1e-6)
        return self.model_.predict(np.hstack([X_capped, X_queue]))


# ============================================================
# SIMULATION AGENT
# ============================================================
class SimulationAgent:
    """
    Multi-hop Global Causal DAG Simulator.

    Workflow:
      1. train()       — build 28-node DAG tu data/raw, fit tat ca co che nhan qua
      2. simulate_intervention(injection_service, delta_pct)
                       — do(injection_service_workload = baseline x (1+delta/100))
                       — tra dict {service: {cpu_change_pct, mem_change_pct, lat_change_pct}}
      3. get_baseline_workload(service)
                       — tra workload baseline de ParserAgent co the dung neu can

    Cau truc DAG (2 tang):
      Tang 1: {svc}_workload -> {svc2}_workload  (theo topology sockshop_agent_graph.json)
      Tang 2: {svc}_workload -> {svc}_cpu / _mem / _latency-50
    """

    def __init__(self, data_dir: str = None, graph_path: str = None):
        if data_dir is None:
            data_dir = os.path.join(BASE_DIR, 'data', 'raw')
        if graph_path is None:
            graph_path = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')

        self.data_dir   = data_dir
        self.graph_path = graph_path

        self.global_model = None
        self.df_baseline  = None
        self.dag          = None
        self._is_trained  = False

        print(f"[SimulationAgent] Graph: {graph_path}")

    # ----------------------------------------------------------
    # TRAIN — build & fit Global DAG
    # ----------------------------------------------------------
    def train(self):
        """
        Xay dung va huan luyen Global 28-node Causal DAG.

        Qua trinh:
          1. Load du lieu da dich vu (load_multi_service_data)
          2. Xay dung DAG theo topology tu JSON
          3. auto.assign_causal_mechanisms
          4. Ghi de domain-aware mechanisms:
             - CPU/Mem: AdditiveNoiseModel(LinearRegression) — tuyen tinh voi workload
             - Latency: AdditiveNoiseModel(QueueingLatencyRegressor) — phi tuyen bao hoa
          5. gcm.fit()
        """
        import json

        print("\n" + "=" * 70)
        print("  [SimulationAgent] XAY DUNG GLOBAL DAG 28-NODE")
        print("  Do(front-end_workload) -> lan truyen qua call chain thuc te")
        print("=" * 70)

        # Load du lieu
        df_data = load_multi_service_data()
        if df_data is None or df_data.empty:
            print("[WARNING] Khong load duoc du lieu da dich vu.")
            self._is_trained = False
            return

        # Load topology
        with open(self.graph_path, 'r', encoding='utf-8') as f:
            graph_json = json.load(f)

        g = nx.DiGraph()

        # Tang 1: Workload -> Workload theo topology thuc
        for edge in graph_json['edges']:
            src, tgt = edge['source'], edge['target']
            if src in SERVICES and tgt in SERVICES:
                g.add_edge(f"{src}_workload", f"{tgt}_workload")

        # Tang 2: Workload -> Metrics noi tai
        for s in SERVICES:
            for metric_col in [f'{s}_cpu', f'{s}_mem', f'{s}_latency-50']:
                if f'{s}_workload' in df_data.columns and metric_col in df_data.columns:
                    g.add_edge(f"{s}_workload", metric_col)

        # Loc chi lay node co trong data
        valid_nodes = [n for n in g.nodes() if n in df_data.columns]
        g_sub = g.subgraph(valid_nodes).copy()
        df_sub = df_data[valid_nodes].dropna()

        if df_sub.empty:
            print("[WARNING] Du lieu sau khi loc bi rong.")
            self._is_trained = False
            return

        # Sample de giam thoi gian train
        df_fit = (df_sub.sample(min(2000, len(df_sub)), random_state=42)
                  if len(df_sub) > 2000 else df_sub)

        print(f"  Nodes trong DAG: {g_sub.number_of_nodes()} | Edges: {g_sub.number_of_edges()}")
        print(f"  Du lieu train: {len(df_fit):,} mau")

        # Fit model
        model = gcm.InvertibleStructuralCausalModel(g_sub)
        gcm.auto.assign_causal_mechanisms(model, df_fit)

        # Ghi de domain-aware mechanisms
        for node in g_sub.nodes():
            if node.endswith('_cpu') or node.endswith('_mem'):
                model.set_causal_mechanism(
                    node,
                    AdditiveNoiseModel(SklearnRegressionModel(LinearRegression()))
                )
            elif node.endswith('_latency-50'):
                model.set_causal_mechanism(
                    node,
                    AdditiveNoiseModel(SklearnRegressionModel(QueueingLatencyRegressor()))
                )

        gcm.fit(model, df_fit)

        self.global_model = model
        self.df_baseline  = df_sub
        self.dag          = g_sub
        self._is_trained  = True

        n_nodes = g_sub.number_of_nodes()
        print(f"\n[OK] Da huan luyen Global DAG ({n_nodes} nodes)")

    # ----------------------------------------------------------
    # SIMULATE — can thiep 1 diem, lay cascade result
    # ----------------------------------------------------------
    def simulate_intervention(
        self,
        injection_service: str,
        injection_delta_pct: float,
        n_samples: int = N_SAMPLES
    ) -> dict:
        """
        Do-calculus: can thiep tai 1 diem duy nhat, Global DAG tu lan truyen.

        Args:
            injection_service:   Service nhan can thiep (thuong la gateway)
            injection_delta_pct: % tang workload tai diem can thiep
            n_samples:           So mau Monte Carlo

        Returns:
            dict: {
              service_name: {
                "cpu_change_pct":     float,
                "mem_change_pct":     float,
                "latency_change_pct": float,
                "n_hops":             int   # so hop tu injection point
              }
            }
        """
        if not self._is_trained:
            print("[WARNING] SimulationAgent chua duoc train. Goi train() truoc.")
            return {}

        injection_col = f"{injection_service}_workload"
        if injection_col not in self.df_baseline.columns:
            print(f"[WARNING] {injection_col} khong co trong du lieu baseline.")
            return {}

        base_wl    = self.df_baseline[injection_col].mean()
        target_wl  = base_wl * (1 + injection_delta_pct / 100)

        print(f"  [Simulation] do({injection_col} = {base_wl:.2f} -> {target_wl:.2f} "
              f"req/s, +{injection_delta_pct}%)")

        samples = gcm.interventional_samples(
            self.global_model,
            interventions={injection_col: lambda x, w=target_wl: w},
            num_samples_to_draw=n_samples
        )

        # Tinh so hop tu injection point
        hops = self._compute_hops(injection_service)

        # Parse results
        results = {}
        for svc in SERVICES:
            entry = {'n_hops': hops.get(svc, -1)}
            for metric_key, col_suffix in [
                ('cpu_change_pct',     '_cpu'),
                ('mem_change_pct',     '_mem'),
                ('latency_change_pct', '_latency-50'),
            ]:
                col = f"{svc}{col_suffix}"
                if col in self.df_baseline.columns and col in samples.columns:
                    base = self.df_baseline[col].mean()
                    pred = samples[col].mean()
                    chg  = (pred - base) / abs(base) * 100 if base != 0 else 0.0
                    entry[metric_key] = round(chg, 2)
                else:
                    entry[metric_key] = None
            results[svc] = entry

        return results

    def _compute_hops(self, injection_service: str) -> dict:
        """Dem so hop tu injection_service den moi service trong DAG."""
        inj_col = f"{injection_service}_workload"
        hops    = {}
        for svc in SERVICES:
            tgt_col = f"{svc}_workload"
            if tgt_col not in self.dag:
                hops[svc] = -1
                continue
            try:
                length = nx.shortest_path_length(self.dag, inj_col, tgt_col)
                hops[svc] = length
            except nx.NetworkXNoPath:
                hops[svc] = -1  # khong co duong lan truyen
            except nx.NodeNotFound:
                hops[svc] = -1
        return hops

    # ----------------------------------------------------------
    # UTILITY
    # ----------------------------------------------------------
    def get_baseline_workload(self, service: str) -> float:
        """Tra workload baseline (req/s) cua service."""
        if not self._is_trained:
            return 0.0
        col = f"{service}_workload"
        return float(self.df_baseline[col].mean()) if col in self.df_baseline else 0.0

    def get_dag_summary(self) -> dict:
        """Tom tat DAG de nhung vao LLM report."""
        if not self._is_trained:
            return {}
        return {
            'n_nodes':   self.dag.number_of_nodes(),
            'n_edges':   self.dag.number_of_edges(),
            'services':  SERVICES,
            'topology':  'Two-tier: workload propagation + local resource impact',
        }
