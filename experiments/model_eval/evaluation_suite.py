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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
import json
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    print("  PHẦN 1: ĐÁNH GIÁ ĐỘ CHÍNH XÁC DỰ BÁO SCM (CHỈ SỐ HỒI QUY CHUẨN)")
    print("  Gold Standard: Train trên LOW workload (67%) -> Test dự báo trên HIGH workload (33%)")
    print("=" * 95)

    eval_results = []
    trained_models = {}

    for metric_name, metric_col, unit, scale in METRICS:
        print(f"\n  [Chỉ số: {metric_name} ({unit})]")
        print(f"  {'Dịch Vụ (Service)':<16} | {'RMSE':>10} | {'MAE':>10} | {'MAPE(%)':>8} | {'SMAPE(%)':>9} | {'NRMSE':>8} | {'R²':>8}")
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
            # Dong bo voi capacity_agent.py::train_fast_path (production) va
            # build_and_train_global_dag (Global DAG): ep LinearRegression(positive=True)
            # thay vi gcm.auto tu chon — thuc nghiem nonlinear_mechanism_trial.py cho
            # thay o dung protocol (train LOW -> test HIGH) linear_pos thang auto_gcm ro
            # ret (MAPE 12.9% vs 20.8%, win 10/21 vs 4/21 cap), va Section "Extrapolation-
            # Sign Failure Mode" cua paper claim rang buoc nay ap dung "uniformly" — truoc
            # ban sua nay claim do khong dung voi RQ1 benchmark.
            model.set_causal_mechanism('Workload', EmpiricalDistribution())
            model.set_causal_mechanism(
                'Target', AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
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

            # --- FULL-RESOLUTION METRIC (khong bucket-averaging) ---
            # Bucket-mean o tren lam mem nhieu va giam con so mau xuong con 8 diem.
            # Voi mo hinh AdditiveNoiseModel, E[Target | do(Workload=w)] = prediction_model.predict(w),
            # nen co the tinh CHINH XAC (khong Monte Carlo) tren TOAN BO diem test tho, khong bucket.
            try:
                mech = model.causal_mechanism('Target')
                yt_full = df_test['Target'].values * scale
                yp_full = mech.prediction_model.predict(df_test[['Workload']].values).ravel() * scale
                mae_full  = mean_absolute_error(yt_full, yp_full)
                rmse_full = np.sqrt(mean_squared_error(yt_full, yp_full))
                mape_full = mape(yt_full, yp_full)
                smape_full = smape(yt_full, yp_full)
                r2_full = r2_score(yt_full, yp_full)
                n_full = len(yt_full)
            except Exception as e:
                mae_full = rmse_full = mape_full = smape_full = r2_full = float('nan')
                n_full = 0
                print(f"    [WARN] Full-resolution metric that bai cho {svc}/{metric_name}: {e}")

            trained_models[(svc, metric_name)] = {
                'model': model,
                'baseline_wl': df_train['Workload'].mean(),
                'baseline_val': df_train['Target'].mean() * scale,
            }

            eval_results.append({
                'evaluation_protocol': 'OOD_Gold_Standard (Train Low -> Test High)',
                'service': svc,
                'metric': metric_name,
                'unit': unit,
                'rmse': round(rmse_v, 4),
                'nrmse': round(nrmse_v, 4) if not np.isnan(nrmse_v) else '',
                'mae': round(mae_v, 4),
                'mape_pct': round(mape_v, 2),
                'smape_pct': round(smape_v, 2) if not np.isnan(smape_v) else '',
                'r2': round(r2_v, 3),
                # Cac chi so full-resolution: tinh tren TOAN BO n_test diem tho (khong bucket).
                # Day la bang chung manh hon vi khong bi lam muot boi trung binh hoa theo bucket.
                'rmse_full_res': round(rmse_full, 4) if not np.isnan(rmse_full) else '',
                'mae_full_res': round(mae_full, 4) if not np.isnan(mae_full) else '',
                'mape_full_res_pct': round(mape_full, 2) if not np.isnan(mape_full) else '',
                'smape_full_res_pct': round(smape_full, 2) if not np.isnan(smape_full) else '',
                'r2_full_res': round(r2_full, 3) if not np.isnan(r2_full) else '',
                'n_train': len(df_train),
                'n_test': len(df_test),
                'n_test_buckets': len(bkts),
                'n_test_full_res': n_full,
                'wl_train_range': f"{df_train['Workload'].min():.1f}-{df_train['Workload'].max():.1f}",
                'wl_test_range': f"{df_test['Workload'].min():.1f}-{df_test['Workload'].max():.1f}",
            })

            print(f"  {svc:<16} | {rmse_v:>10.4f} | {mae_v:>10.4f} | {mape_v:>7.1f}% | {smape_v:>8.1f}% | {nrmse_v:>8.3f} | {r2_v:>8.3f} | full-res MAPE={mape_full:>6.1f}% (n={n_full})")

    return pd.DataFrame(eval_results), trained_models

class QueueingLatencyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model_ = LinearRegression(fit_intercept=True)
        self.capacity_ = None

    def fit(self, X, y):
        X = np.array(X)
        # Ước lượng dung lượng tối đa (Capacity) = Workload lớn nhất * 1.5
        self.capacity_ = np.max(X, axis=0) * 1.5 
        self.capacity_[self.capacity_ == 0] = 1.0 
        
        # Đặc trưng hàng đợi: X / (C - X) tương tự rho / (1 - rho)
        X_queue = X / (self.capacity_ - X + 1e-6)
        X_transformed = np.hstack([X, X_queue])
        
        self.model_.fit(X_transformed, y)
        return self

    def predict(self, X):
        X = np.array(X)
        # Khi ngoại suy (Flash Sale), Workload có thể vượt Capacity,
        # Giới hạn X ở mức 0.99 * C để Latency bùng nổ phi tuyến mà không bị âm hay lỗi chia 0.
        X_capped = np.minimum(X, self.capacity_ * 0.99)
        X_queue = X_capped / (self.capacity_ - X_capped + 1e-6)
        
        X_transformed = np.hstack([X_capped, X_queue])
        return self.model_.predict(X_transformed)


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

    # 2.5. Canh backpressure (caller CPU -> callee CPU) -- DONG BO voi
    # capacity_agent.py::train_accurate_path. Xem comment day du + trich dan
    # 3 script kiem dinh o do; o day chi lap lai dung 3 canh da xac nhan
    # (accuracy + decision-quality + OOD-sign-safety, khong dong loat).
    for caller_cpu, callee_cpu in [('orders_cpu', 'shipping_cpu'), ('orders_cpu', 'carts_cpu'),
                                    ('front-end_cpu', 'user_cpu')]:
        if caller_cpu in df_data.columns and callee_cpu in df_data.columns:
            g.add_edge(caller_cpu, callee_cpu)

    valid_nodes = [n for n in g.nodes() if n in df_data.columns]
    g_sub = g.subgraph(valid_nodes).copy()
    df_sub = df_data[valid_nodes].dropna()

    df_fit = df_sub.sample(min(2000, len(df_sub)), random_state=42) if len(df_sub) > 2000 else df_sub

    model = gcm.InvertibleStructuralCausalModel(g_sub)
    gcm.auto.assign_causal_mechanisms(model, df_fit)
    
    # 3. Ghi đè Domain-Aware Causal Mechanisms để giải quyết Rủi ro Ngoại suy
    #
    # SUA LOI PHAT HIEN 2025-09-08 (2 vong): Ban dau dung LinearRegression thuong cho
    # CPU/Mem, nhung tai do(workload) rat lon (+150%/+300%) mot so node cho delta AM —
    # vo ly ve vat ly (CPU/Memory khong the GIAM khi tai tang). Thu IsotonicRegression cho
    # CPU/Mem thi het loi o do, NHUNG loi van con o TANG 1 (Workload -> Workload giua cac
    # service, vd carts_workload/user_workload co 2 parent) vi cac canh nay VAN dung mechanism
    # tu dong chon boi gcm.auto (co the la mo hinh khong rang buoc dau, vd cay/tuyen tinh he
    # so am do nhieu) — lam workload ha luu GIAM du workload thuong luu tang, keo theo CPU
    # ha luu am theo.
    #
    # Fix triet de: dung LinearRegression(positive=True) (rang buoc he so KHONG AM qua
    # scipy.optimize.nnls) cho TOAN BO canh nhan qua co huong tang don dieu ky vong — ca
    # Workload->Workload (Tang 1) LAN Workload->CPU/Mem (Tang 2). Uu diem so voi Isotonic:
    # ap dung duoc ca khi node co NHIEU parent (vd carts_workload co 2 parent), khong chi
    # 1 bien nhu Isotonic. Danh doi: van la ngoai suy TUYEN TINH khong gioi han bien do (co
    # the qua lon o OOD cuc doan) nhung KHONG THE sai dau — dung day la dieu kien can de
    # dam bao tinh don dieu vat ly xuyen suot toan bo do thi nhan qua 28-node.
    for node in g_sub.nodes():
        if node.endswith('_cpu') or node.endswith('_mem'):
            model.set_causal_mechanism(
                node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))
        elif node.endswith('_latency-50'):
            # Độ trễ tăng phi tuyến theo đường cong bão hòa
            model.set_causal_mechanism(node, AdditiveNoiseModel(SklearnRegressionModel(QueueingLatencyRegressor())))
        elif node.endswith('_workload') and g_sub.in_degree(node) > 0:
            # Canh lan truyen Tang 1 (vd front-end_workload -> orders_workload): workload
            # thuong luu tang khong the lam workload ha luu GIAM.
            model.set_causal_mechanism(
                node, AdditiveNoiseModel(SklearnRegressionModel(LinearRegression(positive=True))))

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

            # PHAT HIEN NGOAI SUY VO LY (extrapolation sanity check): voi do(workload) TANG,
            # CPU/Memory ve nguyen tac vat ly khong the GIAM manh. O cac muc can thiep rat xa
            # mien train (Flash Sale +150%, Black Friday +300%), mechanism LinearRegression
            # cho CPU/Memory co the ngoai suy sai dau (da quan sat: co truong hop am vai
            # tram %). Day la dau hieu mo hinh da vuot qua vung tin cay, KHONG phai ket qua
            # nhan qua that — phai gan co gian ro rang thay vi de lan vao risk_status binh
            # thuong, tranh bi hieu nham la "SCM du bao CPU/Memory giam khi tang tai".
            extrapolation_suspect = (delta_pct > 0) and (c_val < -5.0 or m_val < -5.0 or l_val < -5.0)
            if c_val < -5.0 and delta_pct > 0:
                cpu_chg += " [NGOẠI SUY BẤT THƯỜNG]"
            if m_val < -5.0 and delta_pct > 0:
                mem_chg += " [NGOẠI SUY BẤT THƯỜNG]"
            if l_val < -5.0 and delta_pct > 0:
                lat_chg += " [NGOẠI SUY BẤT THƯỜNG]"

            # LUU Y VE PHUONG PHAP LUAN: cac nguong 30%/15%/100% duoi day la HEURISTIC
            # (chon tron, chua duoc hieu chinh tu du lieu crash/OOM that). Chua co
            # nhan "he thong da thuc su sap do" trong RE2-SS de fit nguong nay tu
            # du lieu. Vi vay risk_status CHI la mot canh bao dinh tinh dua tren quy
            # tac tu dat, KHONG phai ket qua da duoc kiem chung bang su co that —
            # khong nen trich dan nhu "SCM du bao dung crash" trong bao cao khoa hoc
            # neu chua doi chieu voi thoi diem loi that trong RE2-SS (inject_time.txt
            # + cac metric bat thuong sau injection).
            if extrapolation_suspect:
                risk_status = "🚫 NGOẠI SUY BẤT THƯỜNG — KHÔNG ĐÁNG TIN (mô hình vượt vùng tin cậy)"
            elif c_val >= 30.0 or delta_pct >= 100 or l_val >= 100.0:
                risk_status = "❌ CẢNH BÁO (HEURISTIC): NGUY CƠ QUÁ TẢI SỤP ĐỔ (CRITICAL OVERLOAD CRASH)"
            elif l_val >= 30.0 or c_val >= 15.0:
                risk_status = "⚠️ CẢNH BÁO (HEURISTIC): GIẬT LAG MẠNH (SEVERE LATENCY SPIKE)"
            else:
                risk_status = "✅ AN TOÀN (HEURISTIC, chưa đối chiếu sự cố thật)"

            print(f"        {svc:<14} | {cpu_chg:>12} | {mem_chg:>14} | {lat_chg:>14} | {risk_status}")
            sim_rows.append({
                'scenario_name': scenario_name,
                'query': query,
                'request_type': detected_rtype,
                'feature_resource_profile': resource_prof,
                'do_workload_delta_pct': delta_pct,
                'extrapolation_suspect': extrapolation_suspect,
                'service': svc,
                'cpu_change_pct': cpu_chg,
                'mem_change_pct': mem_chg,
                'lat_p50_change_pct': lat_chg,
                'risk_status': risk_status,
                'risk_status_is_validated_against_real_incidents': False,
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
    """LƯU Ý (sau đợt audit RQ2): hàm này báo cáo p-value THÔ, KHÔNG hiệu chỉnh
    đa so sánh và KHÔNG chặn hậu kiểm theo Friedman omnibus. Nó chạy ~12 kiểm
    định trên mỗi hệ thống nên cột "CÓ Ý NGHĨA" ở đây KHÔNG dùng được cho bài
    báo. Số liệu chính thức lấy từ experiments/rq2_statistical_analysis.py
    (Friedman gating + Holm + effect size theo hạng) -> rq2_corrected_statistics.csv.
    Giữ lại hàm này chỉ để đối chiếu lịch sử."""
    print("\n" + "=" * 90)
    print("  KIỂM ĐỊNH THỐNG KÊ (WILCOXON & FRIEDMAN) — BẢN THÔ, KHÔNG HIỆU CHỈNH")
    print("  >> KHÔNG trích cột 'CÓ Ý NGHĨA' vào bài báo. Dùng rq2_statistical_analysis.py.")
    print("=" * 90)

    CSV_PATH_1 = os.path.join(OUT_DIR, 'MODEL_COMPARISON.csv')
    CSV_PATH_2 = os.path.join(OUT_DIR, '05_model_comparison.csv')
    CSV_PATH = CSV_PATH_1 if os.path.exists(CSV_PATH_1) else CSV_PATH_2
    OUT_PATH = os.path.join(OUT_DIR, 'p_value_statistical_test.csv')

    if not os.path.exists(CSV_PATH):
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy chạy model_comparison.py trước.")
        return

    df = pd.read_csv(CSV_PATH)

    # SUA LOI PHUONG PHAP LUAN: khong con gop RMSE tho cua CPU(%), Memory(MB),
    # Socket(count) vao chung mot phep kiem dinh — cac don vi khac nhau lam
    # rank cua Wilcoxon vo nghia. Dung MAPE (chi so % da chuan hoa, so sanh duoc
    # giua cac don vi), chay RIENG cho tung metric + 1 ban tong hop ALL_COMBINED.
    #
    # Chay CA HAI phien ban MAPE de doi chieu ro rang muc do nhay cam voi cach
    # tinh: 'mape_pct' (trung binh tren 8 bucket, lam muot nhieu) va
    # 'mape_full_res_pct' (tren TOAN BO diem test tho, khong lam muot) — ban
    # full-resolution chat che hon nen la can cu chinh khi ket luan trong bai bao.
    baselines = ['LinearReg', 'GradBoost', 'GaussianProcess']

    def paired_values(sub_df, model_a, model_b, value_col):
        """Ghep cap dung theo (service, metric) — tranh lech thu tu hang giua 2 model."""
        a = sub_df[sub_df['model'] == model_a].set_index(['service', 'metric'])[value_col]
        b = sub_df[sub_df['model'] == model_b].set_index(['service', 'metric'])[value_col]
        common = a.index.intersection(b.index)
        return a.loc[common].values, b.loc[common].values, len(common)

    stat_results = []

    for value_col in ['mape_pct', 'mape_full_res_pct']:
        if value_col not in df.columns:
            continue
        valid = df.dropna(subset=[value_col])
        metrics_present = sorted(valid['metric'].unique())

        # (1) Kiem dinh rieng cho tung metric (CPU / Memory / Socket)
        for metric_name in metrics_present:
            sub = valid[valid['metric'] == metric_name]
            for bl in baselines:
                a, b, n = paired_values(sub, 'SCM_Deployed', bl, value_col)
                if n >= 2 and not np.allclose(a, b):
                    w_stat, w_p = stats.wilcoxon(a, b)
                else:
                    w_stat, w_p = float('nan'), float('nan')
                stat_results.append({
                    'comparison': f'SCM_vs_{bl}',
                    'metric': metric_name,
                    'value_col': value_col,
                    'n_pairs': n,
                    'median_diff_SCM_minus_baseline': round(float(np.median(a - b)), 3) if n >= 1 else '',
                    'wilcoxon_stat': round(w_stat, 3) if n >= 2 else '',
                    'p_value': round(w_p, 5) if n >= 2 else '',
                    'significant_p_lt_0.05': bool(w_p < 0.05) if n >= 2 else False,
                })
            # Friedman rieng cho tung metric (chi tinh duoc voi n_service dong nhat qua 4 model)
            piv = sub.pivot_table(index='service', columns='model', values=value_col)
            piv = piv.dropna(subset=['SCM_Deployed'] + baselines)
            if len(piv) >= 3:
                f_stat, f_p = stats.friedmanchisquare(
                    piv['SCM_Deployed'], piv['LinearReg'], piv['GradBoost'], piv['GaussianProcess'])
            else:
                f_stat, f_p = float('nan'), float('nan')
            stat_results.append({
                'comparison': 'Friedman_4_Models',
                'metric': metric_name,
                'value_col': value_col,
                'n_pairs': len(piv),
                'median_diff_SCM_minus_baseline': '',
                'wilcoxon_stat': round(f_stat, 3) if len(piv) >= 3 else '',
                'p_value': round(f_p, 5) if len(piv) >= 3 else '',
                'significant_p_lt_0.05': bool(f_p < 0.05) if len(piv) >= 3 else False,
            })

        # (2) Kiem dinh tong hop tren toan bo (service, metric)
        for bl in baselines:
            a, b, n = paired_values(valid, 'SCM_Deployed', bl, value_col)
            if n >= 2 and not np.allclose(a, b):
                w_stat, w_p = stats.wilcoxon(a, b)
            else:
                w_stat, w_p = float('nan'), float('nan')
            stat_results.append({
                'comparison': f'SCM_vs_{bl}',
                'metric': 'ALL_COMBINED',
                'value_col': value_col,
                'n_pairs': n,
                'median_diff_SCM_minus_baseline': round(float(np.median(a - b)), 3) if n >= 1 else '',
                'wilcoxon_stat': round(w_stat, 3) if n >= 2 else '',
                'p_value': round(w_p, 5) if n >= 2 else '',
                'significant_p_lt_0.05': bool(w_p < 0.05) if n >= 2 else False,
            })

        piv_all = valid.pivot_table(index=['service', 'metric'], columns='model', values=value_col)
        piv_all = piv_all.dropna(subset=['SCM_Deployed'] + baselines)
        if len(piv_all) >= 3:
            f_stat, f_p = stats.friedmanchisquare(
                piv_all['SCM_Deployed'], piv_all['LinearReg'], piv_all['GradBoost'], piv_all['GaussianProcess'])
        else:
            f_stat, f_p = float('nan'), float('nan')
        stat_results.append({
            'comparison': 'Friedman_4_Models',
            'metric': 'ALL_COMBINED',
            'value_col': value_col,
            'n_pairs': len(piv_all),
            'median_diff_SCM_minus_baseline': '',
            'wilcoxon_stat': round(f_stat, 3) if len(piv_all) >= 3 else '',
            'p_value': round(f_p, 5) if len(piv_all) >= 3 else '',
            'significant_p_lt_0.05': bool(f_p < 0.05) if len(piv_all) >= 3 else False,
        })

    df_stat = pd.DataFrame(stat_results)
    df_stat.to_csv(OUT_PATH, index=False)

    print(f"\n  {'Đối Chiếu':<22} | {'Metric':<13} | {'n':>3} | {'Median Δ':>9} | {'Stat':>9} | {'p-value':>10} | {'Ý nghĩa'}")
    print("  " + "-" * 95)
    for value_col, grp in df_stat.groupby('value_col', sort=False):
        print(f"\n  [value_col = {value_col}]")
        for _, r in grp.iterrows():
            sig = "✅ CÓ Ý NGHĨA" if r['significant_p_lt_0.05'] else "❌ KHÔNG"
            pv = r['p_value'] if r['p_value'] != '' else float('nan')
            st = r['wilcoxon_stat'] if r['wilcoxon_stat'] != '' else float('nan')
            md = r['median_diff_SCM_minus_baseline'] if r['median_diff_SCM_minus_baseline'] != '' else float('nan')
            print(f"  {r['comparison']:<22} | {r['metric']:<13} | {r['n_pairs']:>3} | {md:>9} | {st:>9} | {pv:>10} | {sig}")
    print("\n  LƯU Ý: p > 0.05 chỉ có nghĩa là KHÔNG BÁC BỎ được giả thuyết 'không khác biệt' —")
    print("  KHÔNG phải bằng chứng cho thấy hai mô hình 'tương đương'. Muốn khẳng định tương")
    print("  đương cần kiểm định equivalence (TOST), chưa được thực hiện ở đây.")

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

# =============================================================================
# PHẦN 6: RQ4 — GIÁ TRỊ CỦA LAN TRUYỀN TẦNG 1 (WORKLOAD -> WORKLOAD)
# =============================================================================

def _build_workload_graph():
    """Doc topology va tra ve DiGraph cac canh Workload->Workload (Tang 1)."""
    with open(JSON_GRAPH_PATH, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)
    g = nx.DiGraph()
    for edge in graph_json['edges']:
        src, tgt = edge['source'], edge['target']
        if src in SERVICES and tgt in SERVICES:
            g.add_edge(f"{src}_workload", f"{tgt}_workload")
    return g


def _rq4_single_split(df_sub, g, split_ratio):
    """
    Chay 1 lan protocol SCM-vs-Naive (Tang 1 propagation) tren 1 nguong chia
    train/test cu the. Tra ve list dict, moi phan tu la 1 (service, split_ratio).
    Tach rieng ham nay de tai su dung cho ca ban goc (1 nguong 0.67) lan ban
    da-nguong (multi-split) ben duoi, tranh viet trung logic.
    """
    df_sorted = df_sub.sort_values('front-end_workload').reset_index(drop=True)
    split_idx = int(len(df_sorted) * split_ratio)
    df_train, df_test = df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]

    rows = []
    for s in SERVICES:
        node = f"{s}_workload"
        parents = [p for p in g.predecessors(node)] if node in g.nodes() else []
        parents = [p for p in parents if p in df_sub.columns]
        if not parents or node not in df_sub.columns:
            continue  # root node (front-end) hoac thieu du lieu

        reg = LinearRegression(positive=True)
        reg.fit(df_train[parents].values, df_train[node].values)
        scm_pred = reg.predict(df_test[parents].values)

        fe_train_mean = df_train['front-end_workload'].mean()
        node_train_mean = df_train[node].mean()
        fe_test = df_test['front-end_workload'].values
        naive_pred = node_train_mean * (fe_test / fe_train_mean) if fe_train_mean != 0 else np.full_like(fe_test, node_train_mean)

        actual = df_test[node].values

        def _metrics(y_pred):
            return {
                'rmse': float(np.sqrt(mean_squared_error(actual, y_pred))),
                'mae': float(mean_absolute_error(actual, y_pred)),
                'mape_pct': float(mape(actual, y_pred)),
            }

        m_scm, m_naive = _metrics(scm_pred), _metrics(naive_pred)

        rows.append({
            'split_ratio': split_ratio, 'service': s, 'node': node, 'parents': str(parents),
            'n_train': len(df_train), 'n_test': len(df_test),
            'scm_rmse': round(m_scm['rmse'], 4), 'naive_rmse': round(m_naive['rmse'], 4),
            'scm_mape_pct': round(m_scm['mape_pct'], 2), 'naive_mape_pct': round(m_naive['mape_pct'], 2),
            'scm_better_rmse': m_scm['rmse'] < m_naive['rmse'],
            'scm_better_mape': m_scm['mape_pct'] < m_naive['mape_pct'],
        })
    return rows


def run_rq4_propagation_value_test(df_data=None):
    """
    RQ4: Lan truyền workload qua đồ thị phụ thuộc thật (Tầng 1) có chính xác hơn
    giả định "delta đều" (naive: mọi downstream service đổi CÙNG % với front-end)
    hay không?

    Protocol: OOD Gold Standard giống RQ1 — fit Tầng 1 (LinearRegression(positive=True),
    parent thật theo topology trong sockshop_agent_graph.json) trên 67% front-end_workload
    THẤP nhất, test trên 33% CAO nhất — dùng CHÍNH GIÁ TRỊ front-end_workload đo được
    trong tập test (không giả lập do() tổng hợp) để hai phương pháp cùng nhận input,
    rồi so cả hai với s_workload THẬT đo cùng thời điểm.

    LƯU Ý VỀ CỠ MẪU: chỉ có n=6 service không-gốc trong topology SockShop — đây là
    giới hạn CỐ ĐỊNH của topology, không thể tăng bằng cách gộp thêm điểm dữ liệu
    thô (làm vậy sẽ vi phạm giả định độc lập của Wilcoxon — pseudo-replication).
    Muốn có thêm bằng chứng, xem `run_rq4_multisplit_replication()` bên dưới, chạy
    trên NHIỀU cấu hình thực nghiệm độc lập (nhiều ngưỡng chia train/test) thay vì
    giả vờ có nhiều dữ liệu hơn từ cùng 1 cấu hình.

    Output: rq4_propagation_value_test.csv
    """
    print("\n" + "=" * 95)
    print("  RQ4: LAN TRUYỀN QUA TẦNG 1 (WORKLOAD→WORKLOAD) VS GIẢ ĐỊNH DELTA ĐỀU (NAIVE)")
    print("=" * 95)

    if df_data is None:
        df_data = load_multi_service_data()
    if df_data is None or df_data.empty:
        print("  Lỗi: không load được dữ liệu đa dịch vụ.")
        return pd.DataFrame()

    g = _build_workload_graph()
    wl_cols = [f"{s}_workload" for s in SERVICES if f"{s}_workload" in df_data.columns]
    df_sub = df_data[wl_cols].dropna()

    results = _rq4_single_split(df_sub, g, split_ratio=0.67)
    for r in results:
        print(f"  {r['service']:<12} | parents={r['parents']:<45} | SCM MAPE={r['scm_mape_pct']:>7.2f}% | "
              f"Naive MAPE={r['naive_mape_pct']:>7.2f}% | {'SCM thắng' if r['scm_better_mape'] else 'Naive thắng'}")

    df_out = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, 'rq4_propagation_value_test.csv')
    df_out.to_csv(out_path, index=False)

    if len(df_out) >= 2:
        a, b = df_out['scm_mape_pct'].values, df_out['naive_mape_pct'].values
        w_stat, w_p = stats.wilcoxon(a, b) if not np.allclose(a, b) else (float('nan'), float('nan'))
        print(f"\n  Wilcoxon SCM vs Naive (MAPE, n={len(df_out)}): stat={w_stat}, p={w_p:.5f}" if not np.isnan(w_p) else "\n  Wilcoxon: không đủ khác biệt để tính.")
        print(f"  SCM thắng {df_out['scm_better_mape'].sum()}/{len(df_out)} node theo MAPE.")

    print(f"\n  ✅ Đã lưu: {out_path}")
    return df_out


def run_rq4_multisplit_replication(df_data=None, split_ratios=(0.60, 0.65, 0.67, 0.70, 0.75), n_bootstrap=10000, seed=42):
    """
    RQ4 (mo rong tin cay): lap lai DUNG protocol o tren tren NHIEU nguong chia
    train/test khac nhau (60/40 .. 75/25) thay vi chi 1 nguong 67/33 — moi nguong
    la 1 CAU HINH THUC NGHIEM DOC LAP VE THIET KE (diem cat OOD khac nhau), tang so
    quan sat tu 6 len 6 x len(split_ratios) MOT CACH HOP LE, khong phai bang cach
    gop diem du lieu tho (tranh pseudo-replication).

    Hai bang chung duoc bao cao SONG SONG, khong thay the nhau:
      1. Wilcoxon tren toan bo (service x split_ratio) — luu y: cac nguong dung
         chung 1 nguon du lieu goc nen KHONG hoan toan doc lap nhu i.i.d. thuc su;
         bao cao ro gioi han nay thay vi coi la n=30 "sach".
      2. Bootstrap CI 95% tren DUNG 6 chenh lech goc (o nguong 67/33 chinh) —
         khong gia vo co nhieu du lieu hon 6, chi dinh luong do khong chac chan
         THAT SU co duoc tu 6 diem do bang resampling.

    Output: rq4_propagation_value_multisplit.csv
    """
    print("\n" + "=" * 95)
    print(f"  RQ4 (MULTI-SPLIT): LAP LAI TREN {len(split_ratios)} NGUONG CHIA TRAIN/TEST DOC LAP")
    print("=" * 95)

    if df_data is None:
        df_data = load_multi_service_data()
    if df_data is None or df_data.empty:
        print("  Lỗi: không load được dữ liệu đa dịch vụ.")
        return pd.DataFrame(), {}

    g = _build_workload_graph()
    wl_cols = [f"{s}_workload" for s in SERVICES if f"{s}_workload" in df_data.columns]
    df_sub = df_data[wl_cols].dropna()

    all_rows = []
    for ratio in split_ratios:
        rows = _rq4_single_split(df_sub, g, split_ratio=ratio)
        all_rows.extend(rows)
        n_win = sum(1 for r in rows if r['scm_better_mape'])
        print(f"  [split={ratio:.2f}] {len(rows)} service | SCM thắng {n_win}/{len(rows)}")

    df_multi = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, 'rq4_propagation_value_multisplit.csv')
    df_multi.to_csv(out_path, index=False)

    # (1) Wilcoxon tren toan bo (service x split_ratio)
    a, b = df_multi['scm_mape_pct'].values, df_multi['naive_mape_pct'].values
    if len(df_multi) >= 2 and not np.allclose(a, b):
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float('nan'), float('nan')
    n_total = len(df_multi)
    n_win_total = int(df_multi['scm_better_mape'].sum())

    # (2) Bootstrap CI tren DUNG 6 chenh lech goc o nguong chinh (0.67, hoac gan nhat)
    canonical_ratio = min(split_ratios, key=lambda r: abs(r - 0.67))
    df_canon = df_multi[df_multi['split_ratio'] == canonical_ratio]
    diffs = (df_canon['scm_mape_pct'] - df_canon['naive_mape_pct']).values  # am = SCM tot hon
    rng = np.random.RandomState(seed)
    n_orig = len(diffs)
    boot_medians = np.array([
        np.median(rng.choice(diffs, size=n_orig, replace=True)) for _ in range(n_bootstrap)
    ]) if n_orig > 0 else np.array([])
    if len(boot_medians) > 0:
        ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
        pct_boot_favor_scm = float((boot_medians < 0).mean() * 100)
    else:
        ci_low = ci_high = pct_boot_favor_scm = float('nan')

    summary = {
        'n_splits': len(split_ratios),
        'n_total_observations': n_total,
        'n_scm_wins_total': n_win_total,
        'wilcoxon_stat_pooled': round(w_stat, 3) if not np.isnan(w_stat) else None,
        'wilcoxon_p_pooled': round(w_p, 5) if not np.isnan(w_p) else None,
        'canonical_split_ratio': canonical_ratio,
        'n_canonical_services': n_orig,
        'bootstrap_median_diff_ci_low': round(float(ci_low), 3) if not np.isnan(ci_low) else None,
        'bootstrap_median_diff_ci_high': round(float(ci_high), 3) if not np.isnan(ci_high) else None,
        'bootstrap_pct_resamples_favor_scm': round(pct_boot_favor_scm, 1) if not np.isnan(pct_boot_favor_scm) else None,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, 'rq4_multisplit_summary.csv'), index=False)

    print(f"\n  [1] Wilcoxon gop {len(split_ratios)} nguong (n={n_total}, LUU Y: khong hoan toan doc lap): "
          f"stat={w_stat:.2f}, p={w_p:.5f}" if not np.isnan(w_p) else "\n  [1] Wilcoxon: không đủ khác biệt.")
    print(f"      SCM thắng {n_win_total}/{n_total} tổ hợp (service x split_ratio).")
    print(f"  [2] Bootstrap 95% CI cho median(SCM-Naive) tại nguong chinh {canonical_ratio} (n that = {n_orig}):")
    print(f"      [{ci_low:.2f}, {ci_high:.2f}] | {pct_boot_favor_scm:.1f}% số lần resample nghiêng về SCM")
    print(f"\n  ✅ Đã lưu: {out_path}")
    print(f"  ✅ Đã lưu tóm tắt: rq4_multisplit_summary.csv")
    return df_multi, summary


if __name__ == '__main__':
    print("Vui lòng chạy file run_all_experiments.py ở thư mục gốc để chạy toàn bộ suite.")
