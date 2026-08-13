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
from scm_pipeline import load_normal_data, METRICS, SERVICES, mape

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
        print(f"  {'Dịch Vụ (Service)':<16} | {'RMSE':>10} | {'F1-Score':>10} | {'Prec':>8} | {'Recall':>8} | {'PosR%':>7} | {'MAPE(%)':>7}")
        print("  " + "-" * 85)

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
            mape_v = mape(yt, yp)
            r2_v   = r2_score(yt, yp)

            thresh = np.percentile(df['Target'].values * scale, 80)
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
                'f1_score': round(f1_v, 3),
                'precision': round(prec_v, 3),
                'recall': round(rec_v, 3),
                'pos_ratio_pct': round(pos_ratio, 1),
                'mae': round(mae_v, 4),
                'mape_pct': round(mape_v, 2),
                'r2': round(r2_v, 3),
            })

            print(f"  {svc:<16} | {rmse_v:>10.4f} | {f1_v:>10.3f} | {prec_v:>8.3f} | {rec_v:>8.3f} | {pos_ratio:>7.1f}% | {mape_v:>7.1f}%")

    return pd.DataFrame(eval_results), trained_models

def test_new_features_simulation(trained_models):
    print("\n" + "=" * 95)
    print("  PHẦN 2: THỬ NGHIỆM ĐẦU VÀO VÀ MÔ PHỎNG TÍNH NĂNG MỚI (CHƯA CÓ TRÊN SOCKSHOP)")
    print("=" * 95)

    test_queries = [
        ("Tính năng: Áp dụng Promo Code", "Áp mã voucher giảm giá 20% khi thanh toán", "APPLY_PROMO_CODE"),
        ("Tính năng: AI Gợi ý sản phẩm", "Gợi ý các sản phẩm tất thông minh cho tôi", "RECOMMEND_PRODUCTS"),
        ("Tính năng: Theo dõi đơn hàng Real-time", "Xem hành trình giao hàng và vị trí đơn hàng real-time", "TRACK_PACKAGE"),
        ("Tính năng: Đánh giá Review sản phẩm", "Viết nhận xét đánh giá 5 sao cho sản phẩm", "WRITE_PRODUCT_REVIEW"),
        ("Baseline: Đặt hàng tiêu chuẩn", "Đặt hàng mua sản phẩm", "PLACE_ORDER"),
        ("Sự kiện: Siêu Flash Sale (+150% Load)", "Sự kiện Flash Sale giảm giá 90% siêu lớn toàn hệ thống", "PLACE_ORDER"),
    ]

    sim_rows = []
    for scenario_name, query, expected_rtype in test_queries:
        detected_rtype = classify_request(query)
        blast = get_blast_radius(detected_rtype)
        affected_svcs = blast['affected_services']
        delta_pct = blast['expected_delta_pct']
        resource_prof = blast['resource_profile']

        if "Flash Sale" in query:
            delta_pct = 150

        print(f"\n  📝 Truy vấn đầu vào: \"{query}\"")
        print(f"     -> Phân loại: {detected_rtype} (Khớp kỳ vọng: {detected_rtype == expected_rtype})")
        print(f"     -> Mô tả tính năng: {blast['description']}")
        print(f"     -> Blast Radius: {' -> '.join(affected_svcs)}")
        print(f"     -> Can thiệp do(WL) = +{delta_pct}% | Profile: {resource_prof}")
        print(f"        {'Service':<14} | {'CPU Delta':>12} | {'Memory Delta':>14} | {'Lat_p50 Delta':>14} | {'CẢNH BÁO RỦI RO'}")
        print("        " + "-" * 90)

        for svc in affected_svcs:
            cpu_key, mem_key, lat_key = (svc, 'CPU'), (svc, 'Memory'), (svc, 'Latency_p50')
            cpu_chg, mem_chg, lat_chg = "N/A", "N/A", "N/A"
            c_val, m_val, l_val = 0.0, 0.0, 0.0

            if cpu_key in trained_models:
                m_info = trained_models[cpu_key]
                new_wl = m_info['baseline_wl'] * (1 + delta_pct / 100)
                dp = gcm.interventional_samples(m_info['model'], interventions={'Workload': lambda x, w=new_wl: w}, num_samples_to_draw=N_PROJ_STANDARD)
                new_val = dp['Target'].mean() * 1.0
                c_val = (new_val - m_info['baseline_val']) / abs(m_info['baseline_val']) * 100 if m_info['baseline_val'] != 0 else 0
                cpu_chg = f"{c_val:+.1f}%"

            if mem_key in trained_models:
                m_info = trained_models[mem_key]
                new_wl = m_info['baseline_wl'] * (1 + delta_pct / 100)
                dp = gcm.interventional_samples(m_info['model'], interventions={'Workload': lambda x, w=new_wl: w}, num_samples_to_draw=N_PROJ_STANDARD)
                new_val = dp['Target'].mean() * (1/1e6)
                m_val = (new_val - m_info['baseline_val']) / abs(m_info['baseline_val']) * 100 if m_info['baseline_val'] != 0 else 0
                mem_chg = f"{m_val:+.1f}%"

            if lat_key in trained_models:
                m_info = trained_models[lat_key]
                new_wl = m_info['baseline_wl'] * (1 + delta_pct / 100)
                dp = gcm.interventional_samples(m_info['model'], interventions={'Workload': lambda x, w=new_wl: w}, num_samples_to_draw=N_PROJ_STANDARD)
                new_val = dp['Target'].mean() * 1000.0
                l_val = (new_val - m_info['baseline_val']) / abs(m_info['baseline_val']) * 100 if m_info['baseline_val'] != 0 else 0
                lat_chg = f"{l_val:+.1f}%"

            if c_val >= 30.0 or delta_pct >= 100:
                risk_status = "❌ CẢNH BÁO: NGUY CƠ QUÁ TẢI SỤP ĐỔ (CRITICAL OVERLOAD CRASH)"
            elif l_val >= 50.0 or c_val >= 15.0:
                risk_status = "⚠️ CẢNH BÁO: GIẬT LAG MẠNH (SEVERE LATENCY SPIKE)"
            else:
                risk_status = "✅ AN TOÀN (NORMAL)"

            print(f"        {svc:<14} | {cpu_chg:>12} | {mem_chg:>14} | {lat_chg:>14} | {risk_status}")
            sim_rows.append({
                'scenario_name': scenario_name, 'query': query, 'request_type': detected_rtype, 'feature_resource_profile': resource_prof,
                'do_workload_delta_pct': delta_pct, 'service': svc, 'cpu_change_pct': cpu_chg,
                'mem_change_pct': mem_chg, 'lat_p50_change_pct': lat_chg, 'risk_status': risk_status
            })

    return pd.DataFrame(sim_rows)

# =============================================================================
# PHẦN 3: KIỂM THỬ ĐỒ THỊ NHÂN QUẢ 14 NODE (MULTI-NODE GRAPH)
# =============================================================================

def load_multi_service_data():
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
                
                svc_cols = [tc]
                for s in SERVICES:
                    wcol, ccol = f'{s}_workload', f'{s}_cpu'
                    if wcol in cols and ccol in cols:
                        svc_cols.extend([wcol, ccol])
                
                df_run = pd.read_csv(mp, usecols=list(set(svc_cols)))
                df_run = df_run[df_run[tc] < it].drop(columns=[tc]).dropna()
                
                merged_df = df_run if merged_df is None else pd.concat([merged_df, df_run], ignore_index=True)
            except Exception:
                continue
    return merged_df

def test_14_node_causal_graph():
    print("\n" + "=" * 90)
    print("  🚀 KIỂM THỬ TRỰC TIẾP ĐỒ THỊ NHÂN QUẢ 14 NODE KIẾN TRÚC SOCKSHOP")
    print("=" * 90)

    with open(JSON_GRAPH_PATH, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)

    g = nx.DiGraph()
    for edge in graph_json['edges']:
        if edge['source'] in SERVICES and edge['target'] in SERVICES:
            g.add_edge(f"{edge['source']}_cpu", f"{edge['target']}_cpu")
    g.add_edge("front-end_workload", "front-end_cpu")

    df_data = load_multi_service_data()
    if df_data is None or df_data.empty:
        print("Lỗi: Không thể nạp dữ liệu đa dịch vụ.")
        return

    graph_nodes = list(g.nodes())
    valid_cols = [c for c in graph_nodes if c in df_data.columns]
    df_sub = df_data[valid_cols].dropna().head(1000)

    model = gcm.InvertibleStructuralCausalModel(g)
    gcm.auto.assign_causal_mechanisms(model, df_sub)
    gcm.fit(model, df_sub)

    base_wl = df_sub['front-end_workload'].mean()
    target_wl = base_wl * 1.50 

    samples = gcm.interventional_samples(model, interventions={'front-end_workload': lambda x, w=target_wl: w}, num_samples_to_draw=300)

    print(f"\n  📊 Kết Quả Lan Truyền Can Thiệp do() Qua Các Tầng Dịch Vụ 14-Node:")
    print(f"  {'Nút Dịch Vụ (Service Node)':<25} | {'CPU Gốc (%)':>12} | {'CPU Dự Báo do() (%)':>20} | {'Biến Động (%)':>14}")
    print("  " + "-" * 80)

    csv_rows = []
    for node in graph_nodes:
        if node in df_sub.columns and node in samples.columns:
            base_val = df_sub[node].mean()
            pred_val = samples[node].mean()
            chg = ((pred_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
            print(f"  {node:<25} | {base_val:>12.4f} | {pred_val:>20.4f} | {chg:>+13.1f}%")
            csv_rows.append({
                'Service_Node': node,
                'Original_CPU_Pct': round(base_val, 4),
                'Predicted_do_CPU_Pct': round(pred_val, 4),
                'Change_Pct': round(chg, 2)
            })
    
    df_out = pd.DataFrame(csv_rows)
    out_file = os.path.join(OUT_DIR, '14_node_causal_propagation.csv')
    df_out.to_csv(out_file, index=False)
    print(f"\n  ✅ Đã lưu kết quả lan truyền 14-Node tại: {out_file}")

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

def split_quantile(df):
    df_sorted = df.sort_values('Workload').reset_index(drop=True)
    split = int(len(df_sorted) * 0.67)
    return df_sorted.iloc[:split], df_sorted.iloc[split:]

def split_random(df):
    return train_test_split(df, test_size=0.3, random_state=42)

def split_chronological(df):
    split = int(len(df) * 0.70)
    return df.iloc[:split], df.iloc[split:]

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
            thresh = np.percentile(df['Target'].values * scale, 80)
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
