# -*- coding: utf-8 -*-
"""
KIỂM THỬ TRỰC TIẾP ĐỒ THỊ NHÂN QUẢ 14 NODE (MULTI-NODE SYSTEM CAUSAL GRAPH TEST)
================================================================================
Script này chứng minh khả năng dựng và thực thi phép can thiệp do-calculus trực tiếp
trên Đồ Thị Nhân Quả Kiến Trúc 14 Node (sockshop_agent_graph.json) toàn hệ thống.
"""

import os
import sys
import json
import warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
JSON_GRAPH_PATH = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

SERVICES = ['front-end', 'catalogue', 'carts', 'user', 'orders', 'payment', 'shipping']


def load_multi_service_data():
    """Load combined workload data for microservices to build joint dataset."""
    merged_df = None
    for scenario in os.listdir(RAW_DATA_DIR):
        sp = os.path.join(RAW_DATA_DIR, scenario)
        if not os.path.isdir(sp): continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp): continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not os.path.exists(mp) or not os.path.exists(ip): continue
            try:
                with open(ip) as f: it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None: continue
                
                # Load CPU metrics of key services
                svc_cols = [tc]
                for s in SERVICES:
                    wcol = f'{s}_workload'
                    ccol = f'{s}_cpu'
                    if wcol in cols and ccol in cols:
                        svc_cols.extend([wcol, ccol])
                
                df_run = pd.read_csv(mp, usecols=list(set(svc_cols)))
                df_run = df_run[df_run[tc] < it].drop(columns=[tc]).dropna()
                
                if merged_df is None:
                    merged_df = df_run
                else:
                    merged_df = pd.concat([merged_df, df_run], ignore_index=True)
            except Exception:
                continue
    return merged_df


def test_14_node_causal_graph():
    print("=" * 90)
    print("  🚀 KIỂM THỬ TRỰC TIẾP ĐỒ THỊ NHÂN QUẢ 14 NODE KIẾN TRÚC SOCKSHOP")
    print("=" * 90)

    # 1. Đọc file JSON đồ thị 14 node
    with open(JSON_GRAPH_PATH, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)

    print(f"\n  📌 Đã tải đồ thị 14 Node từ sockshop_agent_graph.json:")
    print(f"     -> Tổng số Nút (Nodes): {len(graph_json['nodes'])}")
    print(f"     -> Tổng số Cạnh kết nối (Edges): {len(graph_json['edges'])}")

    # 2. Xây dựng NetworkX DiGraph cho các nút dịch vụ chính
    g = nx.DiGraph()
    for edge in graph_json['edges']:
        src = edge['source']
        tgt = edge['target']
        # Ánh xạ kết nối giữa các dịch vụ chính
        if src in SERVICES and tgt in SERVICES:
            g.add_edge(f"{src}_cpu", f"{tgt}_cpu")

    # Bổ sung nút Workload tác động vào Gateway front-end
    g.add_edge("front-end_workload", "front-end_cpu")

    print(f"\n  🌐 Đã khởi tạo Multi-Node System Causal Graph:")
    print(f"     -> Danh sách cạnh nhân quả liên dịch vụ: {list(g.edges())}")

    # 3. Tải dữ liệu và huấn luyện Multi-Node SCM
    df_data = load_multi_service_data()
    if df_data is None or df_data.empty:
        print("Lỗi: Không thể nạp dữ liệu đa dịch vụ.")
        return

    # Lấy danh sách cột khớp với đồ thị
    graph_nodes = list(g.nodes())
    valid_cols = [c for c in graph_nodes if c in df_data.columns]
    df_sub = df_data[valid_cols].dropna()

    # Giới hạn mẫu để fit nhanh
    df_sub = df_sub.head(1000)

    print(f"\n  🧠 Huấn luyện Multi-Node Structural Causal Model trên {len(df_sub)} mẫu...")
    model = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(model, df_sub)
    gcm.fit(model, df_sub)

    # 4. Thực hiện Phép Can Thiệp do() Liên Dịch Vụ
    base_wl = df_sub['front-end_workload'].mean()
    target_wl = base_wl * 1.50 # Bơm +50% Workload tại Gateway

    print(f"\n  ⚡ Thực hiện phép can thiệp do(front-end_workload = +50% Tải)...")
    samples = gcm.interventional_samples(
        model,
        interventions={'front-end_workload': lambda x, w=target_wl: w},
        num_samples_to_draw=300
    )

    print(f"\n  📊 Kết Quả Lan Truyền Can Thiệp do() Qua Các Tầng Dịch Vụ 14-Node:")
    print(f"  {'Nút Dịch Vụ (Service Node)':<25} | {'CPU Gốc (%)':>12} | {'CPU Dự Báo do() (%)':>20} | {'Biến Động (%)':>14}")
    print("  " + "-" * 80)

    for node in graph_nodes:
        if node in df_sub.columns and node in samples.columns:
            base_val = df_sub[node].mean()
            pred_val = samples[node].mean()
            chg = ((pred_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            print(f"  {node:<25} | {base_val:>12.4f} | {pred_val:>20.4f} | {chg:>+13.1f}%")

    print("\n" + "=" * 90)
    print("  ✅ HOÀN THÀNH KIỂM THỬ ĐỒ THỊ NHÂN QUẢ 14 NODE TOÀN HỆ THỐNG!")
    print("=" * 90)

if __name__ == '__main__':
    test_14_node_causal_graph()
