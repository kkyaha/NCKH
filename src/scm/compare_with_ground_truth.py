# -*- coding: utf-8 -*-
"""
Direct Ground-Truth Matching with RCAEval Raw Data
===================================================
So khớp trực tiếp kết quả dự đoán của SCM với các điểm đo đạc thực tế có sẵn
trong các file CSV gốc của bộ dữ liệu RCAEval khi hệ thống tăng tải tự nhiên.
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import networkx as nx
from dowhy import gcm

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'

def run_ground_truth_comparison(scenario='payment_cpu', run_id='1', service='front-end'):
    csv_path = os.path.join(BASE_DIR, 'data', 'raw', scenario, run_id, 'simple_metrics.csv')
    inject_path = os.path.join(BASE_DIR, 'data', 'raw', scenario, run_id, 'inject_time.txt')
    
    with open(inject_path) as f:
        inject_time = int(f.read().strip())
        
    df_raw = pd.read_csv(csv_path)
    tc = 'imte' if 'imte' in df_raw.columns else 'time'
    
    # Chỉ lấy dữ liệu bình thường của service được chọn
    wl_col = f'{service}_workload'
    cpu_col = f'{service}_cpu'
    mem_col = f'{service}_mem'
    lat_col = f'{service}_latency-50'
    
    df_normal = df_raw[df_raw[tc] < inject_time][[wl_col, cpu_col, mem_col, lat_col]].dropna().copy()
    df_normal.columns = ['Workload', 'CPU', 'Memory', 'Latency']
    df_normal['Memory_MB'] = df_normal['Memory'] / 1e6
    df_normal['Latency_ms'] = df_normal['Latency'] * 1000
    
    # Huấn luyện SCM trên 70% dữ liệu đầu
    split_idx = int(len(df_normal) * 0.7)
    df_train = df_normal.iloc[:split_idx]
    df_test_pool = df_normal.iloc[split_idx:]
    
    # Train SCM cho CPU, Memory, Latency
    def fit_scm(target_col):
        g = nx.DiGraph(); g.add_edge('Workload', target_col)
        m = gcm.InvertibleStructuralCausalModel(g)
        gcm.auto.assign_causal_mechanisms(m, df_train[['Workload', target_col]])
        gcm.fit(m, df_train[['Workload', target_col]])
        return m

    m_cpu = fit_scm('CPU')
    m_mem = fit_scm('Memory_MB')
    m_lat = fit_scm('Latency_ms')
    
    base_wl = df_train['Workload'].mean()
    base_cpu = df_train['CPU'].mean()
    base_mem = df_train['Memory_MB'].mean()
    base_lat = df_train['Latency_ms'].mean()
    
    print("=" * 95)
    print(f"  SO KHỚP TRỰC TIẾP VỚI GROUND-TRUTH CÓ SẴN TRONG BỘ DỮ LIỆU RCAEval")
    print(f"  Kịch bản: {scenario}/run_{run_id} | Service: {service.upper()}")
    print("=" * 95)
    print(f"  MỨC TẢI GỐC (Baseline lúc bình thường):")
    print(f"  -> Workload: {base_wl:.2f} req/s | CPU: {base_cpu:.4f} | RAM: {base_mem:.2f} MB | Latency: {base_lat:.2f} ms\n")
    
    # Tìm các điểm đo đạc thực tế trong tập test khi tải tăng dần (+5%, +10%, +15%, +20%)
    quantiles = [0.2, 0.4, 0.6, 0.8, 0.95]
    
    print(f"{'Tải đo thật trong CSV':<22} | {'Tăng tải':<10} | {'CPU (Thật vs SCM)':<22} | {'RAM (Thật vs SCM)':<22} | {'Lat (Thật vs SCM)':<22}")
    print("-" * 105)
    
    for q in quantiles:
        target_wl = df_test_pool['Workload'].quantile(q)
        matched_samples = df_test_pool[(df_test_pool['Workload'] >= target_wl - 0.3) & 
                                       (df_test_pool['Workload'] <= target_wl + 0.3)]
        if len(matched_samples) == 0: continue
        
        act_wl  = matched_samples['Workload'].mean()
        act_cpu = matched_samples['CPU'].mean()
        act_mem = matched_samples['Memory_MB'].mean()
        act_lat = matched_samples['Latency_ms'].mean()
        
        delta_wl_pct = (act_wl - base_wl) / base_wl * 100
        
        # SCM dự báo tại mức tải thật đó
        dp_cpu = gcm.interventional_samples(m_cpu, interventions={'Workload': lambda x, w=act_wl: w}, num_samples_to_draw=500)['CPU'].mean()
        dp_mem = gcm.interventional_samples(m_mem, interventions={'Workload': lambda x, w=act_wl: w}, num_samples_to_draw=500)['Memory_MB'].mean()
        dp_lat = gcm.interventional_samples(m_lat, interventions={'Workload': lambda x, w=act_wl: w}, num_samples_to_draw=500)['Latency_ms'].mean()
        
        err_cpu = abs(dp_cpu - act_cpu) / act_cpu * 100
        err_mem = abs(dp_mem - act_mem) / act_mem * 100
        err_lat = abs(dp_lat - act_lat) / act_lat * 100
        
        print(f"{act_wl:>6.2f} req/s (n={len(matched_samples):>2} mẫu) | {delta_wl_pct:>+6.1f}%   | "
              f"{act_cpu:>6.4f} vs {dp_cpu:>6.4f} ({err_cpu:>3.1f}%) | "
              f"{act_mem:>5.1f} vs {dp_mem:>5.1f}MB ({err_mem:>3.1f}%) | "
              f"{act_lat:>5.1f} vs {dp_lat:>5.1f}ms ({err_lat:>3.1f}%)")

if __name__ == '__main__':
    run_ground_truth_comparison('payment_cpu', '1', 'front-end')
    print("\n")
    run_ground_truth_comparison('orders_cpu', '1', 'user')
