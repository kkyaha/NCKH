# -*- coding: utf-8 -*-
"""
DATA PROCESSOR & DATASET EXPORTER FOR SCM MICROSERVICES
======================================================
Module chuyên trách xử lý dữ liệu viễn trắc (Telemetry) và quản lý
các tập dữ liệu thực nghiệm phục vụ huấn luyện và kiểm thử mô hình.

Xuất ra 2 file dữ liệu chuẩn hóa:
1. 01_system_telemetry_train_test.csv: Toàn bộ dữ liệu viễn trắc 7 dịch vụ kèm nhãn Train/Test OOD
2. 02_system_test_cases_catalog.csv: Danh mục toàn bộ các trường hợp/kịch bản kiểm thử hệ thống
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)

SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']

# Core Resource Capacity Metrics: (display_name, column_suffix, display_unit, scale_to_unit)
METRICS = [
    ('CPU',        'cpu',        '%',   1.0   ),
    ('Memory',     'mem',        'MB',  1/1e6 ),
    ('Socket',     'socket',     'cnt', 1.0   ),
]

CALL_CHAINS = {
    'GET_CATALOGUE':       ['front-end', 'catalogue'],
    'ADD_TO_CART':         ['front-end', 'catalogue', 'carts'],
    'VIEW_CART':           ['front-end', 'carts'],
    'REGISTER':            ['front-end', 'user'],
    'LOGIN':               ['front-end', 'user'],
    'PLACE_ORDER':         ['front-end', 'user', 'catalogue', 'carts', 'orders', 'payment', 'shipping'],
    'APPLY_PROMO_CODE':    ['front-end', 'carts', 'orders', 'payment'],
    'RECOMMEND_PRODUCTS':  ['front-end', 'user', 'catalogue', 'orders'],
    'TRACK_PACKAGE':       ['front-end', 'orders', 'shipping'],
    'WRITE_PRODUCT_REVIEW':['front-end', 'user', 'catalogue'],
}


# =============================================================================
# 1. HÀM TẢI DỮ LIỆU ĐƠN LẺ & ĐA DỊCH VỤ
# =============================================================================

def load_normal_data(service: str, metric_col: str) -> pd.DataFrame:
    """
    Tải dữ liệu chuỗi thời gian bình thường (trước thời điểm inject fault)
    cho 1 dịch vụ cụ thể và 1 chỉ số mục tiêu.
    """
    dfs = []
    wlc = f'{service}_workload'
    tgc = f'{service}_{metric_col}'

    for scenario in os.listdir(RAW_DATA_DIR):
        sp = os.path.join(RAW_DATA_DIR, scenario)
        if not os.path.isdir(sp): continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp): continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)): continue
            try:
                with open(ip) as f: it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None or wlc not in cols or tgc not in cols: continue

                df_raw = pd.read_csv(mp, usecols=[tc, wlc, tgc])
                df = df_raw[[tc, wlc, tgc]].copy()
                df.columns = [tc, 'Workload', 'Target']
                dfs.append(df[df[tc] < it].drop(columns=[tc]).dropna())
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_multi_service_data() -> pd.DataFrame:
    """
    Tải và gộp dữ liệu viễn trắc của toàn bộ 7 dịch vụ vi mô
    cho cả 4 nhóm chỉ số (Workload, CPU, Memory, Socket).
    """
    merged_df = None
    cols_to_keep = []
    for s in SERVICES:
        cols_to_keep.extend([f'{s}_workload', f'{s}_cpu', f'{s}_mem', f'{s}_socket', f'{s}_latency-50', f'{s}_latency-90', f'{s}_latency-99'])
        
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
                df = pd.read_csv(mp)
                tc = 'imte' if 'imte' in df.columns else ('time' if 'time' in df.columns else None)
                if tc is None: continue
                
                avail_cols = [c for c in cols_to_keep if c in df.columns]
                df_run = df[df[tc] < it][avail_cols].dropna()
                merged_df = df_run if merged_df is None else pd.concat([merged_df, df_run], ignore_index=True)
            except Exception:
                continue
    return merged_df


# =============================================================================
# 2. CÁC GIAO THỨC CHIA DỮ LIỆU HUẤN LUYỆN / KIỂM THỬ (SPLIT PROTOCOLS)
# =============================================================================

def split_quantile(df: pd.DataFrame, ratio: float = 0.67):
    """Protocol A (Gold Standard OOD): Train tải Thấp -> Test tải Cao"""
    df_sorted = df.sort_values('Workload').reset_index(drop=True)
    split_idx = int(len(df_sorted) * ratio)
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:].copy()


def split_random(df: pd.DataFrame, ratio: float = 0.70, seed: int = 42):
    """Protocol B: Chia ngẫu nhiên In-Distribution"""
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * ratio)
    return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:].copy()


def split_chronological(df: pd.DataFrame, ratio: float = 0.70):
    """Protocol C: Chia theo trục thời gian gốc"""
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# =============================================================================
# 3. XUẤT 2 FILE DỮ LIỆU THỰC NGHIỆM CHUẨN HÓA (EXPORTERS)
# =============================================================================

def export_file_1_telemetry_dataset() -> str:
    """
    Xuất File 1: Tập dữ liệu viễn trắc hoàn chỉnh 7 dịch vụ kèm gắn nhãn Train/Test
    theo giao thức OOD Gold Standard (67% Low Workload -> 33% High Workload).
    """
    print("\n[FILE 1] Đang xử lý và xuất Tập dữ liệu Viễn trắc Đa dịch vụ (Telemetry Dataset)...")
    df_multi = load_multi_service_data()
    if df_multi is None or df_multi.empty:
        raise ValueError("Không thể nạp dữ liệu đa dịch vụ.")

    # Sắp xếp theo Workload của cổng front-end để gắn nhãn OOD Train/Test
    df_multi = df_multi.sort_values('front-end_workload').reset_index(drop=True)
    split_idx = int(len(df_multi) * 0.67)
    
    df_multi['split_tag'] = 'Test_Set_High_Workload_OOD'
    df_multi.iloc[:split_idx, df_multi.columns.get_loc('split_tag')] = 'Train_Set_Low_Workload'

    out_file = os.path.join(OUT_DIR, '01_system_telemetry_train_test.csv')
    df_multi.to_csv(out_file, index=False)
    print(f"  ✅ Đã xuất File 1: {out_file} ({len(df_multi):,} dòng, {len(df_multi.columns)} cột)")
    print(f"     - Train Set (Low Load 67%): {split_idx:,} dòng")
    print(f"     - Test Set  (High Load 33%): {len(df_multi) - split_idx:,} dòng")
    return out_file


def export_file_2_test_scenarios_catalog() -> str:
    """
    Xuất File 2: Danh mục toàn bộ các trường hợp/kịch bản thử nghiệm đã dùng để chạy và test hệ thống.
    """
    print("\n[FILE 2] Đang tạo Danh mục Kịch bản Thử nghiệm (Test Cases Catalog)...")
    
    test_cases = [
        # --- NHÓM 1: CÁC LUỒNG NGHIỆP VỤ CƠ BẢN (BASELINE MICROSERVICE FLOWS) ---
        {
            'scenario_id': 'TC-01',
            'scenario_category': '1. Baseline Microservice Flow',
            'scenario_name': 'Xem Danh Mục Sản Phẩm',
            'test_query_prompt': 'Xem danh sách tất cả các sản phẩm tất trong cửa hàng',
            'request_type': 'GET_CATALOGUE',
            'resource_profile': 'cpu-memory',
            'workload_delta_pct': 10,
            'call_chain_services': 'front-end -> catalogue',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Kiểm thử luồng đọc dữ liệu cơ bản (Read-only query)'
        },
        {
            'scenario_id': 'TC-02',
            'scenario_category': '1. Baseline Microservice Flow',
            'scenario_name': 'Thêm Sản Phẩm Vào Giỏ Hàng',
            'test_query_prompt': 'Thêm đôi tất thể thao cổ ngắn vào giỏ hàng của tôi',
            'request_type': 'ADD_TO_CART',
            'resource_profile': 'general',
            'workload_delta_pct': 15,
            'call_chain_services': 'front-end -> catalogue -> carts',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Kiểm thử cập nhật trạng thái giỏ hàng phân tán'
        },
        {
            'scenario_id': 'TC-03',
            'scenario_category': '1. Baseline Microservice Flow',
            'scenario_name': 'Xem Chi Tiết Giỏ Hàng',
            'test_query_prompt': 'Hiển thị các mặt hàng hiện có trong giỏ hàng',
            'request_type': 'VIEW_CART',
            'resource_profile': 'general',
            'workload_delta_pct': 10,
            'call_chain_services': 'front-end -> carts',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Kiểm thử truy vấn session giỏ hàng'
        },
        {
            'scenario_id': 'TC-04',
            'scenario_category': '1. Baseline Microservice Flow',
            'scenario_name': 'Đăng Ký Tài Khoản Mới',
            'test_query_prompt': 'Tạo tài khoản khách hàng mới với email và mật khẩu',
            'request_type': 'REGISTER',
            'resource_profile': 'cpu-heavy',
            'workload_delta_pct': 15,
            'call_chain_services': 'front-end -> user',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Kiểm thử ghi nhận thông tin xác thực người dùng'
        },
        {
            'scenario_id': 'TC-05',
            'scenario_category': '1. Baseline Microservice Flow',
            'scenario_name': 'Quy Trình Đặt Hàng Tiêu Chuẩn',
            'test_query_prompt': 'Đặt hàng mua sản phẩm và thanh toán đơn hàng',
            'request_type': 'PLACE_ORDER',
            'resource_profile': 'cpu-heavy',
            'workload_delta_pct': 25,
            'call_chain_services': 'front-end -> user -> catalogue -> carts -> orders -> payment -> shipping',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Kiểm thử luồng giao dịch liên kết toàn bộ 7 dịch vụ'
        },

        # --- NHÓM 2: CÁC TÍNH NĂNG MỚI GIẢ ĐỊNH (WHAT-IF NEW FEATURES) ---
        {
            'scenario_id': 'TC-06',
            'scenario_category': '2. What-If New Feature Request',
            'scenario_name': 'Áp Dụng Mã Giảm Giá (Promo Code)',
            'test_query_prompt': 'Áp mã voucher giảm giá 20% khi thanh toán giỏ hàng',
            'request_type': 'APPLY_PROMO_CODE',
            'resource_profile': 'cpu-heavy',
            'workload_delta_pct': 20,
            'call_chain_services': 'front-end -> carts -> orders -> payment',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Dự phóng tài nguyên khi thêm tính năng tính toán khuyến mãi'
        },
        {
            'scenario_id': 'TC-07',
            'scenario_category': '2. What-If New Feature Request',
            'scenario_name': 'AI Gợi Ý Sản Phẩm Thông Minh',
            'test_query_prompt': 'Gợi ý các sản phẩm tất thông minh phù hợp cho tôi',
            'request_type': 'RECOMMEND_PRODUCTS',
            'resource_profile': 'cpu-memory',
            'workload_delta_pct': 30,
            'call_chain_services': 'front-end -> user -> catalogue -> orders',
            'expected_risk_status': 'CẢNH BÁO: GIẬT LAG MẠNH (LATENCY SPIKE)',
            'evaluation_purpose': 'Dự phóng tắc nghẽn bộ nhớ/CPU khi chạy thuật toán gợi ý'
        },
        {
            'scenario_id': 'TC-08',
            'scenario_category': '2. What-If New Feature Request',
            'scenario_name': 'Theo Dõi Đơn Hàng Real-Time',
            'test_query_prompt': 'Xem hành trình giao hàng và vị trí đơn hàng real-time qua WebSocket',
            'request_type': 'TRACK_PACKAGE',
            'resource_profile': 'socket-latency',
            'workload_delta_pct': 15,
            'call_chain_services': 'front-end -> orders -> shipping',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Đánh giá ảnh hưởng số lượng kết nối mạng (Socket) liên tục'
        },
        {
            'scenario_id': 'TC-09',
            'scenario_category': '2. What-If New Feature Request',
            'scenario_name': 'Viết Đánh Giá Review Sản Phẩm',
            'test_query_prompt': 'Viết nhận xét đánh giá 5 sao kèm hình ảnh cho sản phẩm',
            'request_type': 'WRITE_PRODUCT_REVIEW',
            'resource_profile': 'disk-memory',
            'workload_delta_pct': 10,
            'call_chain_services': 'front-end -> user -> catalogue',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Đánh giá dung lượng lưu trữ và tải catalogue'
        },

        # --- NHÓM 3: CÁC SỰ KIỆN TĂNG TẢI ĐỘT BIẾN (TRAFFIC SURGES & EXTREME OOD) ---
        {
            'scenario_id': 'TC-10',
            'scenario_category': '3. Traffic Surge & Extreme Stress Event',
            'scenario_name': 'Sự Kiện Mua Sắm Cuối Tuần (+50% Workload)',
            'test_query_prompt': 'Sự kiện mua sắm cuối tuần tăng tải nhẹ trên toàn hệ thống',
            'request_type': 'GET_CATALOGUE',
            'resource_profile': 'cpu',
            'workload_delta_pct': 50,
            'call_chain_services': 'front-end -> catalogue',
            'expected_risk_status': 'AN TOÀN (NORMAL)',
            'evaluation_purpose': 'Kiểm thử sức chịu tải ở mức tăng trưởng trung bình (Medium Load)'
        },
        {
            'scenario_id': 'TC-11',
            'scenario_category': '3. Traffic Surge & Extreme Stress Event',
            'scenario_name': 'Siêu Flash Sale Giảm Giá 90% (+150% Workload)',
            'test_query_prompt': 'Sự kiện Flash Sale giảm giá 90% siêu lớn toàn hệ thống',
            'request_type': 'APPLY_PROMO_CODE',
            'resource_profile': 'cpu-heavy',
            'workload_delta_pct': 150,
            'call_chain_services': 'front-end -> carts -> orders -> payment',
            'expected_risk_status': 'NGUY CƠ QUÁ TẢI SỤP ĐỔ (CRITICAL CRASH)',
            'evaluation_purpose': 'Kiểm thử năng lực dự phóng điểm nghẽn (Bottleneck) khi tải tăng gấp 2.5 lần'
        },
        {
            'scenario_id': 'TC-12',
            'scenario_category': '3. Traffic Surge & Extreme Stress Event',
            'scenario_name': 'Black Friday Siêu Cực Hạn (+300% Workload)',
            'test_query_prompt': 'Sự kiện Black Friday tăng tải cực đại làm sập hệ thống',
            'request_type': 'GET_CATALOGUE',
            'resource_profile': 'cpu',
            'workload_delta_pct': 300,
            'call_chain_services': 'front-end -> catalogue',
            'expected_risk_status': 'NGUY CƠ QUÁ TẢI SỤP ĐỔ (CRITICAL CRASH)',
            'evaluation_purpose': 'Kiểm thử khả năng ngoại suy cực hạn OOD (Tải tăng gấp 4 lần dữ liệu train)'
        },
    ]

    df_catalog = pd.DataFrame(test_cases)
    out_file = os.path.join(OUT_DIR, '02_system_test_cases_catalog.csv')
    df_catalog.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ Đã xuất File 2: {out_file} ({len(df_catalog)} kịch bản thử nghiệm)")
    return out_file


def main():
    print("=" * 85)
    print(" 📦 TRÍCH XUẤT VÀ ĐÓNG GÓI DỮ LIỆU THỰC NGHIỆM CHUẨN HÓA SCM")
    print("=" * 85)
    f1 = export_file_1_telemetry_dataset()
    f2 = export_file_2_test_scenarios_catalog()
    print("\n" + "=" * 85)
    print(" 🎉 HOÀN THÀNH XUẤT 2 FILE DỮ LIỆU!")
    print(f"  1. File Dữ liệu Viễn trắc Train/Test: {f1}")
    print(f"  2. File Danh mục Kịch bản Test Cases: {f2}")
    print("=" * 85)


if __name__ == '__main__':
    main()
