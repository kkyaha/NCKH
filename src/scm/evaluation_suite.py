# -*- coding: utf-8 -*-
"""
HỆ THỐNG ĐÁNH GIÁ TỔNG HỢP (EVALUATION SUITE) CHO SCM
=====================================================
Tích hợp toàn bộ các kịch bản kiểm thử:
1. Đánh giá F1-Score & RMSE.
2. Mô phỏng tính năng mới (Flash Sale, v.v.).
3. Kiểm thử đồ thị nhân quả 14-Node.
4. Kiểm định ý nghĩa thống kê (Wilcoxon/Friedman).
5. Đối chiếu các phương pháp chia tập dữ liệu.
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
from scipy import stats
from dowhy import gcm
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
JSON_GRAPH_PATH = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')

sys.path.append(os.path.join(BASE_DIR, 'src', 'scm'))
from request_router import classify_request, get_blast_radius
from scm_pipeline import mape
from data_processor import (
    load_normal_data, 
    load_multi_service_data, 
    split_quantile, 
    split_random, 
    split_chronological, 
    METRICS, 
    SERVICES
)

def smape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    num = np.abs(yp - yt)
    den = (np.abs(yt) + np.abs(yp)) / 2.0
    m = (den != 0) & np.isfinite(yt) & np.isfinite(yp)
    return np.mean(num[m]/den[m])*100 if m.sum()>0 else float('nan')

N_PROJ_STANDARD = 500
N_PROJ_PROTOCOLS = 300

# =============================================================================
# PHẦN 1 & 2: ĐÁNH GIÁ CHUẨN F1/RMSE & MÔ PHỎNG TÍNH NĂNG MỚI
# =============================================================================

def run_f1_rmse_benchmark():
    print("=" * 95)
    print("  PHẦN 1: ĐÁNH GIÁ MÔ HÌNH SCM THEO CHUẨN F1-SCORE VÀ RMSE")
    print("  Gold Standard: Train trên LOW workload (67%) -> Test dự báo trên HIGH workload (33%)")
    print("=" * 95)

    eval_results = []
    trained_models = {}

    for metric_name, metric_col, unit, scale in METRICS:
        print(f"\n  [Chỉ số: {metric_name} ({unit})]")
        print(f"  {'Dịch Vụ (Service)':<16} | {'RMSE':>10} | {'NRMSE':>8} | {'F1-Score':>10} | {'PosR%':>7} | {'MAPE(%)':>7} | {'SMAPE(%)':>8}")
        print("  " + "-" * 95)

        for svc in SERVICES:
            df = load_normal_data(svc, metric_col)
            if df is None or len(df) < 200:
                continue

            df = df.sort_values('Workload').reset_index(drop=True)
            split = int(len(df) * 0.67)
            df_train = df.iloc[:split]
            df_test  = df.iloc[split:].copy()

            g = nx.DiGraph(); g.add_edge('Workload', 'Target')
            model = gcm.InvertibleStructuralCausalModel(g)
            gcm.auto.assign_causal_mechanisms(model, df_train)
            gcm.fit(model, df_train)

            df_test['bkt'] = pd.qcut(df_test['Workload'], q=min(8, df_test['Workload'].nunique()), duplicates='drop')
            bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

            y_true, y_pred = [], []
            for _, row in bkts.iterrows():
                wlc = row['Workload']
                dp = gcm.interventional_samples(model, interventions={'Workload': lambda x, w=wlc: w}, num_samples_to_draw=N_PROJ_STANDARD)
                y_true.append(row['Target'])
                y_pred.append(dp['Target'].mean())

            yt = np.array(y_true) * scale
            yp = np.array(y_pred) * scale

            mae_v  = mean_absolute_error(yt, yp)
            rmse_v = np.sqrt(mean_squared_error(yt, yp))
            r_range = yt.max() - yt.min()
            nrmse_v = rmse_v / r_range if r_range != 0 else float('nan')
            mape_v = mape(yt, yp)
            smape_v = smape(yt, yp)
            r2_v   = r2_score(yt, yp)

            thresh = np.percentile(df_train['Target'].values * scale, 80)
            yt_bin = (yt >= thresh).astype(int)
            yp_bin = (yp >= thresh).astype(int)

            pos_ratio = yt_bin.mean() * 100.0
            prec_v = precision_score(yt_bin, yp_bin, zero_division=0)
            rec_v  = recall_score(yt_bin, yp_bin, zero_division=0)
            f1_v   = f1_score(yt_bin, yp_bin, zero_division=0)

            trained_models[(svc, metric_name)] = {
                'model': model,
                'baseline_wl': df_train['Workload'].mean(),
                'baseline_val': df_train['Target'].mean() * scale,
            }

            eval_results.append({
                'evaluation_protocol': 'OOD_Gold_Standard (Train Low -> Test High)',
                'risk_threshold': 'P80_Percentile',
                'service': svc,
                'metric': metric_name,
                'unit': unit,
                'rmse': round(rmse_v, 4),
                'nrmse': round(nrmse_v, 4) if not np.isnan(nrmse_v) else '',
                'f1_score': round(f1_v, 3),
                'precision': round(prec_v, 3),
                'recall': round(rec_v, 3),
                'pos_ratio_pct': round(pos_ratio, 1),
                'mae': round(mae_v, 4),
                'mape_pct': round(mape_v, 2),
                'smape_pct': round(smape_v, 2) if not np.isnan(smape_v) else '',
                'r2': round(r2_v, 3),
            })

            print(f"  {svc:<16} | {rmse_v:>10.4f} | {nrmse_v:>8.3f} | {f1_v:>10.3f} | {pos_ratio:>7.1f}% | {mape_v:>7.1f}% | {smape_v:>8.1f}%")

    return pd.DataFrame(eval_results), trained_models

def build_and_train_global_dag(df_data=None):
    if df_data is None:
        df_data = load_multi_service_data()
    if df_data is None or df_data.empty:
        raise ValueError("Không thể load dữ liệu đa dịch vụ.")

    with open(JSON_GRAPH_PATH, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)

    g = nx.DiGraph()
    # 1. Topology edges: Workload -> Workload (Dựa trên kiến trúc gọi API thực tế)
    for edge in graph_json['edges']:
        src, tgt = edge['source'], edge['target']
        if src in SERVICES and tgt in SERVICES:
            g.add_edge(f"{src}_workload", f"{tgt}_workload")

    # 2. Internal metric edges: Workload -> Metrics
    for s in SERVICES:
        for m in [f'{s}_cpu', f'{s}_mem', f'{s}_latency-50']:
            if f'{s}_workload' in df_data.columns and m in df_data.columns:
                g.add_edge(f"{s}_workload", m)

    valid_nodes = [n for n in g.nodes() if n in df_data.columns]
    g_sub = g.subgraph(valid_nodes).copy()
    df_sub = df_data[valid_nodes].dropna()

    df_fit = df_sub.sample(min(2000, len(df_sub)), random_state=42) if len(df_sub) > 2000 else df_sub

    model = gcm.InvertibleStructuralCausalModel(g_sub)
    gcm.auto.assign_causal_mechanisms(model, df_fit)
    gcm.fit(model, df_fit)
    return model, df_sub, g_sub

def test_new_features_simulation(global_model=None, df_sub=None):
    print("\n" + "=" * 95)
    print("  PHẦN 2: THỬ NGHIỆM ĐẦU VÀO VÀ MÔ PHỎNG TÍNH NĂNG MỚI (GLOBAL 28-NODE CAUSAL DAG)")
    print("=" * 95)

    if global_model is None or df_sub is None:
        global_model, df_sub, _ = build_and_train_global_dag()

    test_queries = [
        ("Tính năng: Áp dụng Promo Code", "Áp mã voucher giảm giá 20% khi thanh toán", "APPLY_PROMO_CODE"),
        ("Tính năng: AI Gợi ý sản phẩm", "Gợi ý các sản phẩm tất thông minh cho tôi", "RECOMMEND_PRODUCTS"),
        ("Tính năng: Theo dõi đơn hàng Real-time", "Xem hành trình giao hàng và vị trí đơn hàng real-time", "TRACK_PACKAGE"),
        ("Tính năng: Đánh giá Review sản phẩm", "Viết nhận xét đánh giá 5 sao cho sản phẩm", "WRITE_PRODUCT_REVIEW"),
        ("Baseline: Đặt hàng tiêu chuẩn", "Đặt hàng mua sản phẩm", "PLACE_ORDER"),
        ("Sự kiện: Sale Cuối Tuần (+50% Load)", "Sự kiện mua sắm cuối tuần tăng tải nhẹ", "PLACE_ORDER"),
        ("Sự kiện: Siêu Flash Sale (+150% Load)", "Sự kiện Flash Sale giảm giá 90% siêu lớn toàn hệ thống", "PLACE_ORDER"),
        ("Sự kiện: Black Friday (+300% Load)", "Sự kiện Black Friday tăng tải cực đại làm sập hệ thống", "PLACE_ORDER"),
    ]

    sim_rows = []
    base_fe_wl = df_sub['front-end_workload'].mean()

    for scenario_name, query, expected_rtype in test_queries:
        detected_rtype = classify_request(query)
        blast = get_blast_radius(detected_rtype)
        
        if "Black Friday" in query:
            delta_pct = 300
        elif "Flash Sale" in query:
            delta_pct = 150
        elif "Sale Cuối Tuần" in query:
            delta_pct = 50
        else:
            delta_pct = blast['expected_delta_pct']
            
        resource_prof = blast['resource_profile']

        target_fe_wl = base_fe_wl * (1 + delta_pct / 100)

        # True Interventional Sampling on Global 28-Node DAG
        samples = gcm.interventional_samples(
            global_model, 
            interventions={'front-end_workload': lambda x, w=target_fe_wl: w}, 
            num_samples_to_draw=N_PROJ_STANDARD
        )

        print(f"\n  📝 Truy vấn: \"{query}\" ({scenario_name})")
        print(f"     -> Phân loại: {detected_rtype} | do(front-end_workload) = +{delta_pct}%")
        print(f"     -> Lan truyền tự nhiên qua Đồ thị 28-Node (Global Causal Inference)")
        print(f"        {'Service':<14} | {'CPU Delta':>12} | {'Memory Delta':>14} | {'Lat_p50 Delta':>14} | {'CẢNH BÁO RỦI RO'}")
        print("        " + "-" * 90)

        for svc in SERVICES:
            ccol, mcol, lcol = f'{svc}_cpu', f'{svc}_mem', f'{svc}_latency-50'
            
            c_base = df_sub[ccol].mean() if ccol in df_sub.columns else 1.0
            c_pred = samples[ccol].mean() if ccol in samples.columns else c_base
            c_val = ((c_pred - c_base) / abs(c_base)) * 100 if c_base != 0 else 0

            m_base = df_sub[mcol].mean() if mcol in df_sub.columns else 1.0
            m_pred = samples[mcol].mean() if mcol in samples.columns else m_base
            m_val = ((m_pred - m_base) / abs(m_base)) * 100 if m_base != 0 else 0

            l_base = df_sub[lcol].mean() if lcol in df_sub.columns else 1.0
            l_pred = samples[lcol].mean() if lcol in samples.columns else l_base
            l_val = ((l_pred - l_base) / abs(l_base)) * 100 if l_base != 0 else 0

            cpu_chg = f"{c_val:+.1f}%"
            mem_chg = f"{m_val:+.1f}%"
            lat_chg = f"{l_val:+.1f}%"

            if c_val >= 30.0 or delta_pct >= 100 or l_val >= 100.0:
                risk_status = "❌ CẢNH BÁO: NGUY CƠ QUÁ TẢI SỤP ĐỔ (CRITICAL OVERLOAD CRASH)"
            elif l_val >= 30.0 or c_val >= 15.0:
                risk_status = "⚠️ CẢNH BÁO: GIẬT LAG MẠNH (SEVERE LATENCY SPIKE)"
            else:
                risk_status = "✅ AN TOÀN (NORMAL)"

            print(f"        {svc:<14} | {cpu_chg:>12} | {mem_chg:>14} | {lat_chg:>14} | {risk_status}")
            sim_rows.append({
                'scenario_name': scenario_name,
                'query': query,
                'request_type': detected_rtype,
                'feature_resource_profile': resource_prof,
                'do_workload_delta_pct': delta_pct,
                'service': svc,
                'cpu_change_pct': cpu_chg,
                'mem_change_pct': mem_chg,
                'lat_p50_change_pct': lat_chg,
                'risk_status': risk_status
            })

    return pd.DataFrame(sim_rows)

def test_14_node_causal_graph(global_model=None, df_sub=None):
    print("\n" + "=" * 90)
    print("  🚀 KIỂM THỬ LAN TRUYỀN NHÂN QUẢ 2 TẦNG (7 WORKLOAD + 7 CPU = 14 NODES)")
    print("=" * 90)

    if global_model is None or df_sub is None:
        global_model, df_sub, _ = build_and_train_global_dag()

    base_wl = df_sub['front-end_workload'].mean()
    target_wl = base_wl * 1.50 

    samples = gcm.interventional_samples(
        global_model, 
        interventions={'front-end_workload': lambda x, w=target_wl: w}, 
        num_samples_to_draw=300
    )

    print(f"\n  📊 Kết Quả Lan Truyền Can Thiệp do(front-end_workload = +50%) Qua 2 Tầng Cấu Trúc:")
    print(f"  {'Tầng & Nút Dịch Vụ':<30} | {'Giá Trị Gốc':>12} | {'Dự Báo do()':>15} | {'Biến Động (%)':>14}")
    print("  " + "-" * 80)

    csv_rows = []
    print("  [TẦNG 1: LAN TRUYỀN WORKLOAD GIỮA CÁC DỊCH VỤ]")
    for s in SERVICES:
        col = f"{s}_workload"
        if col in df_sub.columns and col in samples.columns:
            base_val = df_sub[col].mean()
            pred_val = samples[col].mean()
            chg = ((pred_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            print(f"    Workload: {s:<18} | {base_val:>12.2f} | {pred_val:>15.2f} | {chg:>+13.1f}%")
            csv_rows.append({
                'Layer': 'Tier_1_Workload_Propagation',
                'Service': s,
                'Node': col,
                'Original_Value': round(base_val, 4),
                'Predicted_do_Value': round(pred_val, 4),
                'Change_Pct': round(chg, 2)
            })

    print("  [TẦNG 2: TÁC ĐỘNG TẢI NỘI TẠI LÊN CPU CỤC BỘ]")
    for s in SERVICES:
        col = f"{s}_cpu"
        if col in df_sub.columns and col in samples.columns:
            base_val = df_sub[col].mean()
            pred_val = samples[col].mean()
            chg = ((pred_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            print(f"    CPU (%):  {s:<18} | {base_val:>12.4f} | {pred_val:>15.4f} | {chg:>+13.1f}%")
            csv_rows.append({
                'Layer': 'Tier_2_Local_CPU_Impact',
                'Service': s,
                'Node': col,
                'Original_Value': round(base_val, 4),
                'Predicted_do_Value': round(pred_val, 4),
                'Change_Pct': round(chg, 2)
            })

    df_out = pd.DataFrame(csv_rows)
    out_file = os.path.join(OUT_DIR, '14_node_causal_propagation.csv')
    df_out.to_csv(out_file, index=False)
    print(f"\n  ✅ Đã lưu kết quả lan truyền 14-Node (7 Workload + 7 CPU) chuẩn xác tại: {out_file}")

# =============================================================================
# PHẦN 4: KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ
# =============================================================================

def run_statistical_significance():
    print("\n" + "=" * 90)
    print("  CHẠY KIỂM ĐỊNH THỐNG KÊ (WILCOXON & FRIEDMAN TESTS) CHO BÀI BÁO Q1")
    print("=" * 90)

    CSV_PATH_1 = os.path.join(OUT_DIR, 'MODEL_COMPARISON.csv')
    CSV_PATH_2 = os.path.join(OUT_DIR, '05_model_comparison.csv')
    CSV_PATH = CSV_PATH_1 if os.path.exists(CSV_PATH_1) else CSV_PATH_2
    OUT_PATH = os.path.join(OUT_DIR, 'p_value_statistical_test.csv')

    if not os.path.exists(CSV_PATH):
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy chạy model_comparison.py trước.")
        return

    df = pd.read_csv(CSV_PATH)
    valid = df.dropna(subset=['rmse', 'mape_pct', 'f1_score'])

    scm_rmse = valid[valid['model'] == 'SCM_DoWhy']['rmse'].values
    lr_rmse  = valid[valid['model'] == 'LinearReg']['rmse'].values
    gb_rmse  = valid[valid['model'] == 'GradBoost']['rmse'].values
    gp_rmse  = valid[valid['model'] == 'GaussianProcess']['rmse'].values

    w_lr_stat, w_lr_p = stats.wilcoxon(scm_rmse, lr_rmse) if len(scm_rmse) == len(lr_rmse) else (0, 1)
    w_gb_stat, w_gb_p = stats.wilcoxon(scm_rmse, gb_rmse) if len(scm_rmse) == len(gb_rmse) else (0, 1)
    w_gp_stat, w_gp_p = stats.wilcoxon(scm_rmse, gp_rmse) if len(scm_rmse) == len(gp_rmse) else (0, 1)

    min_len = min(len(scm_rmse), len(lr_rmse), len(gb_rmse), len(gp_rmse))
    f_stat, f_p = stats.friedmanchisquare(scm_rmse[:min_len], lr_rmse[:min_len], gb_rmse[:min_len], gp_rmse[:min_len]) if min_len > 0 else (0, 1)

    stat_results = [
        {'comparison': 'SCM_vs_LinearReg', 'metric': 'RMSE', 'wilcoxon_stat': round(w_lr_stat, 3), 'p_value': round(w_lr_p, 5), 'significant_p_lt_0.05': w_lr_p < 0.05},
        {'comparison': 'SCM_vs_GradBoost', 'metric': 'RMSE', 'wilcoxon_stat': round(w_gb_stat, 3), 'p_value': round(w_gb_p, 5), 'significant_p_lt_0.05': w_gb_p < 0.05},
        {'comparison': 'SCM_vs_GaussProc', 'metric': 'RMSE', 'wilcoxon_stat': round(w_gp_stat, 3), 'p_value': round(w_gp_p, 5), 'significant_p_lt_0.05': w_gp_p < 0.05},
        {'comparison': 'Friedman_4_Models_Overall', 'metric': 'RMSE', 'wilcoxon_stat': round(f_stat, 3), 'p_value': round(f_p, 5), 'significant_p_lt_0.05': f_p < 0.05},
    ]

    df_stat = pd.DataFrame(stat_results)
    df_stat.to_csv(OUT_PATH, index=False)

    print(f"\n  {'Đối Chiếu (Comparison)':<30} | {'Wilcoxon Stat':>14} | {'p-value':>10} | {'Ý Nghĩa (p < 0.05)'}")
    print("  " + "-" * 75)
    for _, r in df_stat.iterrows():
        sig = "✅ CÓ Ý NGHĨA THỐNG KÊ" if r['significant_p_lt_0.05'] else "❌ KHÔNG CÓ Ý NGHĨA"
        print(f"  {r['comparison']:<30} | {r['wilcoxon_stat']:>14.3f} | {r['p_value']:>10.5f} | {sig}")

# =============================================================================
# PHẦN 5: ĐỐI CHIẾU CÁC PHƯƠNG PHÁP CHIA TẬP TEST (ALTERNATIVE PROTOCOLS)
# =============================================================================

def run_protocol_evaluation(protocol_name, split_func):
    print(f"\n  📌 CHẠY THỰC NGHIỆM: {protocol_name}")
    print(f"  {'Metric':<10} | {'RMSE':>10} | {'MAPE (%)':>10} | {'F1-Score':>10} | {'Precision':>10} | {'Recall':>10}")
    
    results = []
    for metric_name, metric_col, _, scale in METRICS:
        if metric_name not in ['CPU', 'Memory']: continue # Only test CPU and Memory for speed
        rmse_list, mape_list, f1_list, prec_list, rec_list = [], [], [], [], []

        for svc in SERVICES:
            df = load_normal_data(svc, metric_col)
            if df is None or len(df) < 200: continue
            df_train, df_test = split_func(df)

            g = nx.DiGraph(); g.add_edge('Workload', 'Target')
            model = gcm.InvertibleStructuralCausalModel(g)
            gcm.auto.assign_causal_mechanisms(model, df_train)
            gcm.fit(model, df_train)

            df_test['bkt'] = pd.qcut(df_test['Workload'], q=min(8, df_test['Workload'].nunique()), duplicates='drop')
            bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

            y_true, y_pred = [], []
            for _, row in bkts.iterrows():
                wlc = row['Workload']
                dp = gcm.interventional_samples(model, interventions={'Workload': lambda x, w=wlc: w}, num_samples_to_draw=N_PROJ_PROTOCOLS)
                y_true.append(row['Target'])
                y_pred.append(dp['Target'].mean())

            yt, yp = np.array(y_true) * scale, np.array(y_pred) * scale
            thresh = np.percentile(df_train['Target'].values * scale, 80)
            yt_bin, yp_bin = (yt >= thresh).astype(int), (yp >= thresh).astype(int)

            rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
            mape_list.append(mape(yt, yp))
            f1_list.append(f1_score(yt_bin, yp_bin, zero_division=0))
            prec_list.append(precision_score(yt_bin, yp_bin, zero_division=0))
            rec_list.append(recall_score(yt_bin, yp_bin, zero_division=0))

        results.append({
            'protocol': protocol_name, 'metric': metric_name,
            'avg_rmse': round(np.nanmean(rmse_list), 4), 'avg_mape_pct': round(np.nanmean(mape_list), 2),
            'avg_f1': round(np.nanmean(f1_list), 3), 'avg_precision': round(np.nanmean(prec_list), 3),
            'avg_recall': round(np.nanmean(rec_list), 3),
        })
        print(f"  {metric_name:<10} | {results[-1]['avg_rmse']:>10.4f} | {results[-1]['avg_mape_pct']:>9.1f}% | {results[-1]['avg_f1']:>10.3f} | {results[-1]['avg_precision']:>10.3f} | {results[-1]['avg_recall']:>10.3f}")

    return pd.DataFrame(results)

def compare_all_protocols():
    print("\n" + "=" * 90)
    print(" 🧪 THỬ NGHIỆM ĐỐI CHIẾU 3 CÁCH CHIA TẬP TEST KHÁC NHAU (ALTERNATIVE PROTOCOLS)")
    print("=" * 90)
    res_a = run_protocol_evaluation("Protocol A: Quantile Split (Train Low -> Test High)", split_quantile)
    res_b = run_protocol_evaluation("Protocol B: Random 70/30 Split (In-Distribution)", split_random)
    res_c = run_protocol_evaluation("Protocol C: Chronological Split (Temporal 70/30)", split_chronological)

    df_all = pd.concat([res_a, res_b, res_c], ignore_index=True)
    out_path = os.path.join(OUT_DIR, 'ALTERNATIVE_PROTOCOLS_COMPARISON.csv')
    df_all.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✅ Đã lưu kết quả tại: {out_path}")

if __name__ == '__main__':
    print("Vui lòng chạy file run_all_experiments.py ở thư mục gốc để chạy toàn bộ suite.")
