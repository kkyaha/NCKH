# -*- coding: utf-8 -*-
"""
MASTER EXPERIMENT RUNNER - HỆ THỐNG KIỂM THỬ SCM DỰ BÁO NĂNG LƯỢNG & RỦI RO
===========================================================================
File thực thi tổng hợp chạy toàn bộ các bài thực nghiệm chuẩn NCKH Q1:
1. Đánh giá độ chính xác SCM (RMSE, F1-Score, Precision, Recall) với ngưỡng P80.
2. Benchmark đối chiếu 4 mô hình (LinearReg, GradBoost, GaussProc, SCM).
3. Kiểm định ý nghĩa thống kê Wilcoxon Signed-Rank Test (p-value < 0.05).
4. Kiểm thử Đồ Thị Nhân Quả 14 Node Toàn Hệ Thống (sockshop_agent_graph.json).
5. Phân loại Tiếng Việt & Mô phỏng can thiệp do(WL) cho tính năng MỚI + Flash Sale (+150% WL).
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
sys.path.insert(0, BASE_DIR)

from src.scm.evaluation_suite import (
    run_f1_rmse_benchmark,
    test_new_features_simulation,
    test_14_node_causal_graph,
    run_statistical_significance
)
from src.scm.model_comparison import main as run_model_comparison


def run_full_suite():
    print("=" * 95)
    print(" 🚀 HỆ THỐNG KIỂM THỬ TỔNG HỢP NCKH SCM & DO-CALCULUS (MASTER EXPERIMENT SUITE)")
    print("=" * 95)

    # BƯỚC 1: ĐÁNH GIÁ CHUẨN F1/RMSE & RECALL/PRECISION VỚI NGƯỠNG P80
    print("\n[BƯỚC 1/4] Đang chạy đánh giá mô hình SCM với ngưỡng Percentile P80...")
    df_eval, trained_models = run_f1_rmse_benchmark()
    df_eval.to_csv(r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project\data\processed\scm_results\test_f1_rmse_evaluation.csv', index=False)

    # BƯỚC 2: MÔ PHỎNG TÍNH NĂNG MỚI VÀ KỊCH BẢN FLASH SALE TẢI CỰC ĐẠI
    print("\n[BƯỚC 2/4] Đang mô phỏng các truy vấn Tiếng Việt & kịch bản Flash Sale (+150% WL)...")
    df_sim = test_new_features_simulation(trained_models)
    df_sim.to_csv(r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project\data\processed\scm_results\test_new_features_simulation.csv', index=False)

    # BƯỚC 3: KIỂM THỬ TRỰC TIẾP ĐỒ THỊ NHÂN QUẢ 14 NODE KIẾN TRÚC SOCKSHOP
    print("\n[BƯỚC 3/4] Đang kiểm thử can thiệp do() liên dịch vụ trên Đồ Thị 14 Node...")
    test_14_node_causal_graph()

    # BƯỚC 4: BENCHMARK ĐỐI CHIẾU 4 MÔ HÌNH
    print("\n[BƯỚC 4/5] Đang chạy benchmark đối chiếu 4 mô hình (LinearReg, GradBoost, GaussProc, SCM)...")
    run_model_comparison()

    # BƯỚC 5: KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ P-VALUE (WILCOXON SIGNED-RANK TEST)
    print("\n[BƯỚC 5/5] Đang tính toán kiểm định ý nghĩa thống kê p-value vs Baseline models...")
    run_statistical_significance()

    print("\n" + "=" * 95)
    print(" 🎉 HOÀN THÀNH TOÀN BỘ SUITE KIỂM THỬ TỔNG HỢP!")
    print(" 📊 Kết quả đã được quy tụ gọn gàng về thư mục: data/processed/scm_results/")
    print("    - EXECUTIVE_PROOF_DASHBOARD.csv (Bảng chứng minh Excel 3 Tầng)")
    print("    - MASTER_SUMMARY.md            (Master Dashboard Markdown)")
    print("    - MODEL_COMPARISON.csv         (Data đối chiếu 4 mô hình)")
    print("=" * 95)


if __name__ == '__main__':
    run_full_suite()
