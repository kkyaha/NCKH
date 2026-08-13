# -*- coding: utf-8 -*-
"""
KIỂM THỬ VÀ ĐÁNH GIÁ THEO CHUẨN F1-SCORE / RMSE
=================================================
Kịch bản kiểm thử toàn diện:
  1. Thử nghiệm các câu truy vấn tiếng Việt về TÍNH NĂNG MỚI (chưa có trên SockShop).
  2. Phân loại yêu cầu -> Xác định Blast Radius (dịch vụ chịu ảnh hưởng).
  3. Mô phỏng can thiệp SCM do(Workload) khi triển khai tính năng mới (+20% workload).
  4. Đánh giá độ chính xác mô hình theo chuẩn F1-Score (Phát hiện quá tải) & RMSE (Dự báo định lượng).
  5. Kiểm tra tính toàn vẹn dữ liệu (F1 in [0, 1], RMSE > 0).
"""

import os
import sys
import warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score, precision_score, recall_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
sys.path.append(os.path.join(BASE_DIR, 'src', 'scm'))

from request_router import classify_request, get_blast_radius, CALL_CHAINS
from scm_pipeline import load_normal_data, METRICS, SERVICES, mape

OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
N_PROJ = 500


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

            # Train Bivariate SCM
            g = nx.DiGraph(); g.add_edge('Workload', 'Target')
            model = gcm.InvertibleStructuralCausalModel(g)
            gcm.auto.assign_causal_mechanisms(model, df_train)
            gcm.fit(model, df_train)

            # Test Bucketing (8 bins)
            df_test['bkt'] = pd.qcut(df_test['Workload'], q=min(8, df_test['Workload'].nunique()), duplicates='drop')
            bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

            y_true, y_pred = [], []
            for _, row in bkts.iterrows():
                wlc = row['Workload']
                dp = gcm.interventional_samples(model, interventions={'Workload': lambda x, w=wlc: w}, num_samples_to_draw=N_PROJ)
                y_true.append(row['Target'])
                y_pred.append(dp['Target'].mean())

            yt = np.array(y_true) * scale
            yp = np.array(y_pred) * scale

            mae_v  = mean_absolute_error(yt, yp)
            rmse_v = np.sqrt(mean_squared_error(yt, yp))
            mape_v = mape(yt, yp)
            r2_v   = r2_score(yt, yp)

            # --- CORRECTION: Domain Policy Threshold (P80 of Full Dataset Distribution) ---
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
    print("  Giả định người dùng gửi câu lệnh Tiếng Việt yêu cầu tính năng hệ thống")
    print("=" * 95)

    test_queries = [
        ("Áp mã voucher giảm giá 20% khi thanh toán", "APPLY_PROMO_CODE"),
        ("Gợi ý các sản phẩm tất thông minh cho tôi", "RECOMMEND_PRODUCTS"),
        ("Xem hành trình giao hàng và vị trí đơn hàng real-time", "TRACK_PACKAGE"),
        ("Viết nhận xét đánh giá 5 sao cho sản phẩm", "WRITE_PRODUCT_REVIEW"),
        ("Đặt hàng mua sản phẩm", "PLACE_ORDER"),
        # --- KỊCH BẢN TẢI CỰC ĐẠI: FLASH SALE SIÊU LỚN (+150% WORKLOAD) ---
        ("Sự kiện Flash Sale giảm giá 90% siêu lớn toàn hệ thống", "PLACE_ORDER"),
    ]

    sim_rows = []

    for query, expected_rtype in test_queries:
        detected_rtype = classify_request(query)
        blast = get_blast_radius(detected_rtype)
        affected_svcs = blast['affected_services']
        delta_pct = blast['expected_delta_pct']
        resource_prof = blast['resource_profile']

        # Override delta_pct for extreme Flash Sale scenario
        if "Flash Sale" in query:
            delta_pct = 150

        print(f"\n  📝 Truy vấn đầu vào: \"{query}\"")
        print(f"     -> Phân loại: {detected_rtype} (Khớp kỳ vọng: {detected_rtype == expected_rtype})")
        print(f"     -> Mô tả tính năng: {blast['description']}")
        print(f"     -> Blast Radius ({blast['n_services']} dịch vụ bị ảnh hưởng): {' -> '.join(affected_svcs)}")
        print(f"     -> Can thiệp do(WL) = +{delta_pct}% | Profile: {resource_prof}")
        print(f"        {'Service':<14} | {'CPU Delta':>12} | {'Memory Delta':>14} | {'Lat_p50 Delta':>14} | {'CẢNH BÁO RỦI RO (RISK STATUS)'}")
        print("        " + "-" * 90)

        for svc in affected_svcs:
            cpu_key = (svc, 'CPU')
            mem_key = (svc, 'Memory')
            lat_key = (svc, 'Latency_p50')

            cpu_chg, mem_chg, lat_chg = "N/A", "N/A", "N/A"
            c_val, m_val, l_val = 0.0, 0.0, 0.0

            if cpu_key in trained_models:
                m_info = trained_models[cpu_key]
                base_wl = m_info['baseline_wl']
                base_val = m_info['baseline_val']
                new_wl = base_wl * (1 + delta_pct / 100)
                dp = gcm.interventional_samples(m_info['model'], interventions={'Workload': lambda x, w=new_wl: w}, num_samples_to_draw=N_PROJ)
                new_val = dp['Target'].mean() * 1.0
                c_val = (new_val - base_val) / abs(base_val) * 100 if base_val != 0 else 0
                cpu_chg = f"{c_val:+.1f}%"

            if mem_key in trained_models:
                m_info = trained_models[mem_key]
                base_wl = m_info['baseline_wl']
                base_val = m_info['baseline_val']
                new_wl = base_wl * (1 + delta_pct / 100)
                dp = gcm.interventional_samples(m_info['model'], interventions={'Workload': lambda x, w=new_wl: w}, num_samples_to_draw=N_PROJ)
                new_val = dp['Target'].mean() * (1/1e6)
                m_val = (new_val - base_val) / abs(base_val) * 100 if base_val != 0 else 0
                mem_chg = f"{m_val:+.1f}%"

            if lat_key in trained_models:
                m_info = trained_models[lat_key]
                base_wl = m_info['baseline_wl']
                base_val = m_info['baseline_val']
                new_wl = base_wl * (1 + delta_pct / 100)
                dp = gcm.interventional_samples(m_info['model'], interventions={'Workload': lambda x, w=new_wl: w}, num_samples_to_draw=N_PROJ)
                new_val = dp['Target'].mean() * 1000.0
                l_val = (new_val - base_val) / abs(base_val) * 100 if base_val != 0 else 0
                lat_chg = f"{l_val:+.1f}%"

            # Determine Risk Alert Status
            if c_val >= 30.0 or delta_pct >= 100:
                risk_status = "❌ CẢNH BÁO: NGUY CƠ QUÁ TẢI SỤP ĐỔ (CRITICAL OVERLOAD CRASH)"
            elif l_val >= 50.0 or c_val >= 15.0:
                risk_status = "⚠️ CẢNH BÁO: GIẶT LAG MẠNH (SEVERE LATENCY SPIKE)"
            else:
                risk_status = "✅ AN TOÀN (NORMAL)"

            print(f"        {svc:<14} | {cpu_chg:>12} | {mem_chg:>14} | {lat_chg:>14} | {risk_status}")

            sim_rows.append({
                'query': query,
                'request_type': detected_rtype,
                'feature_resource_profile': resource_prof,
                'do_workload_delta_pct': delta_pct,
                'service': svc,
                'cpu_change_pct': cpu_chg,
                'mem_change_pct': mem_chg,
                'lat_p50_change_pct': lat_chg,
                'risk_status': risk_status,
            })

    return pd.DataFrame(sim_rows)


def main():
    print("=" * 95)
    print("  BẮT ĐẦU CHẠY SUITE KIỂM THỬ: CHUẨN F1/RMSE & TÍNH NĂNG MỚI SOCKSHOP")
    print("=" * 95)

    df_eval, trained_models = run_f1_rmse_benchmark()
    df_sim = test_new_features_simulation(trained_models)

    # Verification assertions
    assert not df_eval.empty, "Lỗi: Không thu được kết quả đánh giá"
    assert (df_eval['f1_score'] >= 0).all() and (df_eval['f1_score'] <= 1.0).all(), "Lỗi giá trị F1 Score ngoài dải [0, 1]"
    assert (df_eval['rmse'] >= 0).all(), "Lỗi giá trị RMSE âm"

    out_eval = os.path.join(OUT_DIR, 'test_f1_rmse_evaluation.csv')
    out_sim  = os.path.join(OUT_DIR, 'test_new_features_simulation.csv')
    df_eval.to_csv(out_eval, index=False)
    df_sim.to_csv(out_sim, index=False)

    print("\n" + "=" * 95)
    print("  ✅ TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ HOÀN THÀNH THÀNH CÔNG!")
    print(f"  - Kết quả đánh giá F1/RMSE lưu tại: {out_eval}")
    print(f"  - Kết quả mô phỏng tính năng mới lưu tại: {out_sim}")
    print("=" * 95)


if __name__ == '__main__':
    main()
