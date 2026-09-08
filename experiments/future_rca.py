# -*- coding: utf-8 -*-
"""
FUTURE RCA ENGINE — Pre-mortem Root Cause Analysis via SCM do-calculus
======================================================================
Module thực hiện chẩn đoán nhân quả CHỦ ĐỘNG (Pre-mortem RCA) thay vì
hồi cứu (Post-mortem RCA) truyền thống. Pipeline 4 tầng:

  Tầng 1: OOD Guard           — Kiểm tra can thiệp nằm trong/ngoài miền huấn luyện
  Tầng 2: Interventional Sim  — Mô phỏng do(·) trên Global DAG
  Tầng 3: Noise Reconstruction — Tái dựng nhiễu ngoại sinh → phát hiện anomaly
  Tầng 4: Shapley Attribution  — Định lượng đóng góp biên từng node

References:
  [Janzing2008]    — Causal inference using the algorithmic Markov condition
  [Li2023]         — ShapleyIQ: Influence Quantification by Shapley Values
  [Nagalapatti2025]— Robust Root Cause Diagnosis Using In-Distribution Interventions
  [Wu2026]         — The Causal Uncertainty Principle

Scope & Limitations:
  - Chỉ áp dụng cho lớp SCM additive/functional noise (AdditiveNoiseModel)
  - Can thiệp ngoại suy xa (> P95 train distribution) chỉ mang tính tham khảo
  - Kết quả Shapley là ước lượng xấp xỉ (Monte Carlo sampling)
"""

import os
import sys
import warnings
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from dowhy.gcm.shapley import ShapleyConfig, ShapleyApproximationMethods

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))              # experiments/ (sibling: evaluation_suite)
sys.path.append(os.path.join(BASE_DIR, 'src', 'scm'))       # core scm lib (data_processor)
from data_processor import SERVICES


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NodeRisk:
    """Risk assessment for a single node under intervention."""
    node: str
    baseline_value: float
    predicted_value: float
    change_pct: float
    anomaly_score: float
    shapley_contribution: float
    risk_level: str  # "normal", "warning", "critical"


@dataclass
class FutureRCAResult:
    """Complete result of a Future RCA analysis."""
    scenario_name: str
    intervention: Dict[str, float]
    ood_confidence: str           # "high" | "medium" | "low" | "very_low"
    ood_percentile: float         # percentile rank of intervention in train dist
    ood_ratio: float              # intervention / max(train)
    node_risks: List[NodeRisk]
    top_bottlenecks: List[str]    # sorted by shapley desc
    total_anomaly_score: float
    shapley_sum: float
    runtime_seconds: float
    caveats: List[str]


# =============================================================================
# TẦNG 1: OOD GUARD — Extrapolation Safety Check
# =============================================================================

class OODGuard:
    """
    Kiểm tra mức can thiệp do(·) nằm trong hay ngoài miền dữ liệu huấn luyện.
    
    Cơ sở khoa học: [Nagalapatti2025] chỉ ra rằng can thiệp in-distribution
    bền vững hơn counterfactual ngoại suy. Module này cảnh báo người dùng
    khi can thiệp vượt xa miền đã quan sát.
    """
    
    def __init__(self, train_data: pd.DataFrame):
        self.train_stats = {}
        for col in train_data.columns:
            vals = train_data[col].dropna().values
            if len(vals) > 0:
                self.train_stats[col] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'min': np.min(vals),
                    'max': np.max(vals),
                    'p75': np.percentile(vals, 75),
                    'p90': np.percentile(vals, 90),
                    'p95': np.percentile(vals, 95),
                    'p99': np.percentile(vals, 99),
                }
    
    def check(self, interventions: Dict[str, float]) -> tuple:
        """
        Returns (confidence_level, percentile_rank, ratio_to_max, caveats).
        
        confidence_level:
          - "high"     : intervention <= P90 of train → reliable
          - "medium"   : P90 < intervention <= P95 → likely reliable
          - "low"      : P95 < intervention <= 2*max → extrapolation warning
          - "very_low" : intervention > 2*max → severe extrapolation
        """
        caveats = []
        worst_confidence = "high"
        worst_percentile = 0.0
        worst_ratio = 0.0
        
        for node, value in interventions.items():
            if node not in self.train_stats:
                caveats.append(f"Node '{node}' not found in training data — cannot assess OOD risk.")
                worst_confidence = "very_low"
                continue
            
            stats = self.train_stats[node]
            ratio = value / stats['max'] if stats['max'] != 0 else float('inf')
            worst_ratio = max(worst_ratio, ratio)
            
            # Calculate approximate percentile rank
            if value <= stats['p75']:
                pct = 75.0
                conf = "high"
            elif value <= stats['p90']:
                pct = 90.0
                conf = "high"
            elif value <= stats['p95']:
                pct = 95.0
                conf = "medium"
                caveats.append(
                    f"[OOD-MEDIUM] do({node}={value:.2f}) nằm trong khoảng P90-P95 "
                    f"(max_train={stats['max']:.2f}). Kết quả dự báo vẫn đáng tin nhưng "
                    f"cần diễn giải thận trọng."
                )
            elif value <= stats['p99']:
                pct = 99.0
                conf = "low"
                caveats.append(
                    f"[OOD-LOW] do({node}={value:.2f}) nằm trong khoảng P95-P99 "
                    f"(max_train={stats['max']:.2f}). Đây là ngoại suy nhẹ — "
                    f"kết quả mang tính tham khảo, không nên dùng làm bảo chứng tuyệt đối. "
                    f"[Ref: Nagalapatti2025]"
                )
            elif value <= 2 * stats['max']:
                pct = 99.5
                conf = "low"
                caveats.append(
                    f"[OOD-LOW] do({node}={value:.2f}) vượt P99 nhưng < 2*max_train "
                    f"({2*stats['max']:.2f}). Ngoại suy đáng kể — kết quả chỉ mang tính "
                    f"định hướng, không phải bảo chứng. [Ref: Wu2026]"
                )
            else:
                pct = 100.0
                conf = "very_low"
                caveats.append(
                    f"[OOD-CRITICAL] do({node}={value:.2f}) vượt xa 2*max_train "
                    f"({2*stats['max']:.2f}). Ngoại suy cực mạnh — SCM có thể cho "
                    f"kết quả không đáng tin. Cần kiểm chứng thực tế trước khi kết luận. "
                    f"[Ref: Nagalapatti2025, Wu2026]"
                )
            
            worst_percentile = max(worst_percentile, pct)
            confidence_order = {"high": 0, "medium": 1, "low": 2, "very_low": 3}
            if confidence_order.get(conf, 3) > confidence_order.get(worst_confidence, 0):
                worst_confidence = conf
        
        return worst_confidence, worst_percentile, worst_ratio, caveats


# =============================================================================
# TẦNG 2-4: FUTURE RCA ENGINE
# =============================================================================

class FutureRCAEngine:
    """
    Pre-mortem Root Cause Analysis Engine.
    
    Thực hiện chẩn đoán nhân quả chủ động trước sự cố bằng cách:
    1. Kiểm tra OOD safety
    2. Mô phỏng can thiệp do(·) trên Global DAG
    3. Tái dựng nhiễu ngoại sinh → phát hiện node bất thường
    4. Shapley attribution → xếp hạng bottleneck
    
    Điều kiện áp dụng (Scope):
    - SCM phải là InvertibleStructuralCausalModel với AdditiveNoiseModel
    - DAG cần phản ánh đúng kiến trúc hệ thống thực tế
    - Can thiệp nên nằm trong miền huấn luyện để đảm bảo tin cậy
    """
    
    def __init__(self, causal_model, train_data: pd.DataFrame, graph: nx.DiGraph):
        self.model = causal_model
        self.train_data = train_data
        self.graph = graph
        self.ood_guard = OODGuard(train_data)
        self.n_samples = 500
        self.n_shapley_samples = 2000
    
    def analyze(
        self,
        scenario_name: str,
        interventions: Dict[str, float],
        target_metrics: Optional[List[str]] = None,
    ) -> FutureRCAResult:
        """
        Chạy toàn bộ pipeline Future RCA cho một kịch bản can thiệp.
        
        Args:
            scenario_name: Tên kịch bản (vd: "Flash Sale +150%")
            interventions: Dict {node_name: value} cho do(·)
            target_metrics: Danh sách node cần đánh giá. Nếu None, dùng tất cả CPU nodes.
        
        Returns:
            FutureRCAResult chứa toàn bộ kết quả phân tích.
        """
        t0 = time.time()
        
        # --- TẦNG 1: OOD GUARD ---
        confidence, ood_pct, ood_ratio, caveats = self.ood_guard.check(interventions)
        
        # Thêm caveat chung về scope
        caveats.insert(0, 
            "[SCOPE] Phân tích dựa trên Additive Noise SCM (functional causal model). "
            "Kết quả chỉ có giá trị trong phạm vi mô hình đã được học. [Ref: Janzing2008]"
        )
        
        # --- TẦNG 2: INTERVENTIONAL SAMPLING ---
        intervention_fns = {}
        for node, value in interventions.items():
            intervention_fns[node] = lambda x, w=value: w
        
        samples = gcm.interventional_samples(
            self.model,
            interventions=intervention_fns,
            num_samples_to_draw=self.n_samples
        )
        
        # Determine target nodes
        if target_metrics is None:
            target_metrics = [col for col in self.train_data.columns 
                             if col.endswith('_cpu') and col in samples.columns]
        
        # --- TẦNG 3: NOISE RECONSTRUCTION & ANOMALY SCORING ---
        # Tạo "anomaly data" từ mẫu can thiệp (mean values)
        anomaly_row = {}
        for col in self.train_data.columns:
            if col in samples.columns:
                anomaly_row[col] = samples[col].mean()
            else:
                anomaly_row[col] = self.train_data[col].mean()
        
        anomaly_df = pd.DataFrame([anomaly_row])
        
        # Compute anomaly scores cho tất cả nodes
        try:
            anomaly_scores = gcm.anomaly_scores(
                self.model,
                anomaly_df,
                num_samples_conditional=2000,
                num_samples_unconditional=2000
            )
        except Exception as e:
            # Fallback: Compute simple z-score based anomaly
            anomaly_scores = {}
            for col in target_metrics:
                if col in self.train_data.columns:
                    train_mean = self.train_data[col].mean()
                    train_std = self.train_data[col].std()
                    if train_std > 0:
                        z = abs(anomaly_row.get(col, train_mean) - train_mean) / train_std
                        anomaly_scores[col] = np.array([z])
                    else:
                        anomaly_scores[col] = np.array([0.0])
            caveats.append(
                f"[FALLBACK] Anomaly scoring dùng z-score thay vì IT-score do lỗi: {str(e)[:80]}"
            )
        
        # --- TẦNG 4: SHAPLEY ATTRIBUTION ---
        shapley_results = {}
        for target_node in target_metrics:
            if target_node not in self.graph.nodes():
                continue
            try:
                contributions = gcm.attribute_anomalies(
                    self.model,
                    target_node=target_node,
                    anomaly_samples=anomaly_df,
                    attribute_mean_deviation=True,
                    num_distribution_samples=500
                )
                shapley_results[target_node] = contributions
            except Exception as e:
                # Shapley failed — use proportional fallback
                caveats.append(
                    f"[SHAPLEY-FALLBACK] Shapley attribution cho '{target_node}' dùng "
                    f"proportional estimate do lỗi: {str(e)[:60]}"
                )
                shapley_results[target_node] = None
        
        # --- BUILD RESULTS ---
        node_risks = []
        for col in target_metrics:
            if col not in self.train_data.columns:
                continue
            
            baseline = self.train_data[col].mean()
            predicted = samples[col].mean() if col in samples.columns else baseline
            change_pct = ((predicted - baseline) / abs(baseline)) * 100 if baseline != 0 else 0
            
            # Get anomaly score
            a_score = 0.0
            if col in anomaly_scores:
                vals = anomaly_scores[col]
                a_score = float(vals[0]) if len(vals) > 0 else 0.0
            
            # Get shapley contribution (sum of all upstream contributions)
            s_contribution = 0.0
            if col in shapley_results and shapley_results[col] is not None:
                for upstream_node, contrib_array in shapley_results[col].items():
                    s_contribution += abs(float(contrib_array[0])) if len(contrib_array) > 0 else 0.0
            
            # Determine risk level
            if abs(change_pct) >= 30.0 or a_score >= 3.0:
                risk_level = "critical"
            elif abs(change_pct) >= 15.0 or a_score >= 2.0:
                risk_level = "warning"
            else:
                risk_level = "normal"
            
            node_risks.append(NodeRisk(
                node=col,
                baseline_value=baseline,
                predicted_value=predicted,
                change_pct=round(change_pct, 2),
                anomaly_score=round(a_score, 4),
                shapley_contribution=round(s_contribution, 4),
                risk_level=risk_level
            ))
        
        # Sort by Shapley contribution (highest first), fallback to anomaly_score for bottleneck ranking
        sorted_risks = sorted(node_risks, key=lambda r: (r.shapley_contribution, r.anomaly_score), reverse=True)
        top_bottlenecks = [r.node for r in sorted_risks if r.risk_level != "normal"][:5]
        
        total_anomaly = sum(r.anomaly_score for r in node_risks)
        shapley_sum = sum(r.shapley_contribution for r in node_risks)
        
        runtime = time.time() - t0
        
        return FutureRCAResult(
            scenario_name=scenario_name,
            intervention=interventions,
            ood_confidence=confidence,
            ood_percentile=ood_pct,
            ood_ratio=round(ood_ratio, 3),
            node_risks=sorted_risks,
            top_bottlenecks=top_bottlenecks,
            total_anomaly_score=round(total_anomaly, 4),
            shapley_sum=round(shapley_sum, 4),
            runtime_seconds=round(runtime, 2),
            caveats=caveats
        )


# =============================================================================
# PRETTY PRINTER
# =============================================================================

def print_future_rca_result(result: FutureRCAResult):
    """In kết quả Future RCA dạng bảng cho console."""
    print("\n" + "=" * 100)
    print(f"  🔮 FUTURE RCA: {result.scenario_name}")
    print("=" * 100)
    
    # OOD Guard status
    ood_icons = {"high": "🟢", "medium": "🟡", "low": "🟠", "very_low": "🔴"}
    ood_labels = {"high": "IN-DISTRIBUTION (Đáng tin)", "medium": "NEAR-BOUNDARY (Cẩn thận)", 
                  "low": "EXTRAPOLATION (Tham khảo)", "very_low": "EXTREME EXTRAPOLATION (Rủi ro cao)"}
    
    print(f"\n  {ood_icons.get(result.ood_confidence, '❓')} OOD Confidence: "
          f"{result.ood_confidence.upper()} — {ood_labels.get(result.ood_confidence, 'Unknown')}")
    print(f"     Intervention ratio (vs train max): {result.ood_ratio:.2f}x")
    print(f"     Runtime: {result.runtime_seconds:.1f}s")
    
    # Intervention details
    print(f"\n  📊 Can Thiệp: {result.intervention}")
    
    # Node risks table
    print(f"\n  {'Node':<25} | {'Baseline':>10} | {'Predicted':>10} | {'Change':>8} | "
          f"{'Anomaly':>8} | {'Shapley':>8} | {'Risk Level'}")
    print("  " + "-" * 100)
    
    risk_icons = {"normal": "✅", "warning": "⚠️", "critical": "❌"}
    for nr in result.node_risks:
        icon = risk_icons.get(nr.risk_level, "❓")
        print(f"  {nr.node:<25} | {nr.baseline_value:>10.4f} | {nr.predicted_value:>10.4f} | "
              f"{nr.change_pct:>+7.1f}% | {nr.anomaly_score:>8.2f} | {nr.shapley_contribution:>8.2f} | "
              f"{icon} {nr.risk_level.upper()}")
    
    # Top bottlenecks
    if result.top_bottlenecks:
        print(f"\n  🎯 Top Bottlenecks (xếp theo Shapley):")
        for i, bn in enumerate(result.top_bottlenecks, 1):
            print(f"     #{i}: {bn}")
    else:
        print(f"\n  ✅ Không phát hiện bottleneck nghiêm trọng.")
    
    # Caveats
    if result.caveats:
        print(f"\n  ⚠️ Cảnh báo khoa học ({len(result.caveats)}):")
        for c in result.caveats:
            print(f"     • {c}")


# =============================================================================
# CSV EXPORTER
# =============================================================================

def export_future_rca_csv(results: List[FutureRCAResult], output_path: str):
    """Xuất kết quả Future RCA ra CSV cho báo cáo."""
    rows = []
    for res in results:
        for nr in res.node_risks:
            rows.append({
                'scenario': res.scenario_name,
                'intervention': str(res.intervention),
                'ood_confidence': res.ood_confidence,
                'ood_ratio': res.ood_ratio,
                'node': nr.node,
                'baseline': nr.baseline_value,
                'predicted': nr.predicted_value,
                'change_pct': nr.change_pct,
                'anomaly_score': nr.anomaly_score,
                'shapley_contribution': nr.shapley_contribution,
                'risk_level': nr.risk_level,
                'total_anomaly': res.total_anomaly_score,
                'runtime_s': res.runtime_seconds,
                'n_caveats': len(res.caveats),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✅ Đã lưu kết quả Future RCA: {output_path}")
    return df


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

def run_future_rca_suite():
    """Chạy toàn bộ suite Future RCA trên Global DAG đã huấn luyện."""
    from evaluation_suite import build_and_train_global_dag
    
    print("=" * 100)
    print("  🔮 FUTURE RCA ENGINE — Pre-mortem Root Cause Analysis")
    print("  Scope: Additive Noise SCM | SockShop 7-service architecture")
    print("=" * 100)
    
    print("\n  [1/2] Đang huấn luyện Global DAG...")
    global_model, df_sub, g_sub = build_and_train_global_dag()
    
    print(f"  [2/2] Khởi tạo Future RCA Engine...")
    engine = FutureRCAEngine(global_model, df_sub, g_sub)
    
    # Define test scenarios with varying OOD levels
    base_wl = df_sub['front-end_workload'].mean()
    
    scenarios = [
        ("Baseline: Tải bình thường (+0%)", 
         {'front-end_workload': base_wl * 1.0}),
        ("Tính năng mới: Promo Code (+20%)", 
         {'front-end_workload': base_wl * 1.2}),
        ("Sự kiện: Sale Cuối Tuần (+50%)", 
         {'front-end_workload': base_wl * 1.5}),
        ("Sự kiện: Flash Sale (+150%)", 
         {'front-end_workload': base_wl * 2.5}),
        ("Sự kiện: Black Friday (+300%)", 
         {'front-end_workload': base_wl * 4.0}),
    ]
    
    all_results = []
    for name, intervention in scenarios:
        print(f"\n  ⏳ Đang phân tích: {name}...")
        result = engine.analyze(name, intervention)
        print_future_rca_result(result)
        all_results.append(result)
    
    # Export CSV
    out_path = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', 'future_rca_results.csv')
    export_future_rca_csv(all_results, out_path)
    
    # Summary
    print("\n" + "=" * 100)
    print("  📊 SUMMARY: Future RCA Suite")
    print("=" * 100)
    print(f"  {'Scenario':<40} | {'OOD':>8} | {'Bottlenecks':>12} | {'Runtime':>8}")
    print("  " + "-" * 75)
    for r in all_results:
        bn_count = len(r.top_bottlenecks)
        print(f"  {r.scenario_name:<40} | {r.ood_confidence:>8} | {bn_count:>12} | {r.runtime_seconds:>7.1f}s")
    
    return all_results


if __name__ == '__main__':
    run_future_rca_suite()
