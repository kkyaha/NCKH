# -*- coding: utf-8 -*-
"""
MASTER EXPERIMENT RUNNER - HỆ THỐNG KIỂM THỬ SCM DỰ BÁO NĂNG LƯỢNG & RỦI RO
===========================================================================
File thực thi tổng hợp chạy toàn bộ các bài thực nghiệm chuẩn NCKH:
1. Đánh giá độ chính xác SCM (RMSE, MAE, MAPE, SMAPE, R²) - bucket-average + full-resolution.
2. Mô phỏng can thiệp do(WL) cho tính năng MỚI + Flash Sale (+150% WL) trên Global 28-Node DAG.
3. Kiểm thử Đồ Thị Nhân Quả 14 Node Toàn Hệ Thống (sockshop_agent_graph.json).
4. Benchmark đối chiếu 4 mô hình (LinearReg, GradBoost, GaussProc, SCM).
5. Kiểm định ý nghĩa thống kê Wilcoxon/Friedman theo TỪNG metric (mape_pct, không gộp đơn vị).
6. So khớp trực tiếp SCM với ground-truth thật trên toàn bộ 90 run RE2-SS (không bucket).
7. Benchmark & ablation LLM Parser với LLM THẬT (RQ3) — mặc định KHÔNG dùng bộ giả lập.

LƯU Ý: các bước 1, 5, 6 tạo ra 3 nguồn bằng chứng ĐỘC LẬP về độ chính xác SCM
(bucket OOD, kiểm định thống kê theo %, và so khớp điểm-that không bucket) — nên
đối chiếu cả 3 khi viết kết luận, không chỉ trích một con số đẹp nhất.
"""

import os
import sys
import warnings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

from experiments.evaluation_suite import (
    run_f1_rmse_benchmark,
    build_and_train_global_dag,
    test_new_features_simulation,
    test_14_node_causal_graph,
    run_statistical_significance,
    run_rq4_propagation_value_test
)
from experiments.model_comparison import main as run_model_comparison
from experiments.parser_benchmark_suite import run_parser_benchmark
from experiments.compare_with_ground_truth import run_ground_truth_comparison_all


def run_full_suite():
    print("=" * 95)
    print(" 🚀 HỆ THỐNG KIỂM THỬ TỔNG HỢP NCKH SCM & DO-CALCULUS (MASTER EXPERIMENT SUITE)")
    print("=" * 95)

    # BƯỚC 1: ĐÁNH GIÁ ĐỘ CHÍNH XÁC SCM (CHỈ SỐ HỒI QUY CHUẨN: MAPE, RMSE, MAE, SMAPE, R²)
    print("\n[BƯỚC 1/7] Đang chạy đánh giá mô hình SCM (chỉ số hồi quy chuẩn)...")
    df_eval, trained_models = run_f1_rmse_benchmark()
    df_eval.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', 'test_f1_rmse_evaluation.csv'), index=False)

    # HUẤN LUYỆN SIÊU ĐỒ THỊ 28-NODE DUY NHẤT (GLOBAL 2-TIER DAG)
    print("\n[HUẤN LUYỆN] Đang dựng và huấn luyện Siêu Đồ Thị Nhân Quả 28-Node (Global 2-Tier DAG)...")
    global_model, df_sub, _ = build_and_train_global_dag()

    # BƯỚC 2: MÔ PHỎNG TÍNH NĂNG MỚI VÀ KỊCH BẢN FLASH SALE TẢI CỰC ĐẠI (GLOBAL 28-NODE DAG)
    print("\n[BƯỚC 2/7] Đang mô phỏng trên Global 28-Node DAG (Tiếng Việt & Flash Sale +150% WL)...")
    df_sim = test_new_features_simulation(global_model, df_sub)
    df_sim.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', 'test_new_features_simulation.csv'), index=False)

    # BƯỚC 3: KIỂM THỬ TRỰC TIẾP ĐỒ THỊ NHÂN QUẢ 14 NODE (7 WORKLOAD + 7 CPU)
    print("\n[BƯỚC 3/7] Đang kiểm thử can thiệp do() liên dịch vụ trên 14 Node (7 Workload + 7 CPU)...")
    test_14_node_causal_graph(global_model, df_sub)

    # BƯỚC 4: BENCHMARK ĐỐI CHIẾU 4 MÔ HÌNH
    print("\n[BƯỚC 4/7] Đang chạy benchmark đối chiếu 4 mô hình (LinearReg, GradBoost, GaussProc, SCM)...")
    run_model_comparison()

    # BƯỚC 5: KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ P-VALUE (WILCOXON SIGNED-RANK TEST)
    print("\n[BƯỚC 5/7] Đang tính toán kiểm định ý nghĩa thống kê p-value vs Baseline models...")
    run_statistical_significance()

    # BƯỚC 6: SO KHỚP TRỰC TIẾP VỚI GROUND-TRUTH THẬT (90 scenario/run, không bucket-average)
    # Đây là bằng chứng độc lập mạnh nhất: train/test tách theo thời gian TRONG TỪNG run,
    # không dùng lại tập train/test của benchmark chính ở BƯỚC 1/4.
    print("\n[BƯỚC 6/7] Đang so khớp SCM với ground-truth thật trên toàn bộ 90 run RE2-SS...")
    run_ground_truth_comparison_all()

    # RQ4: Gia tri cua lan truyen Tang 1 (Workload->Workload) so voi gia dinh delta deu
    print("\n[RQ4] Đang kiểm tra giá trị của lan truyền qua đồ thị Tầng 1 (vs naive delta đều)...")
    run_rq4_propagation_value_test()

    # BƯỚC 7: BENCHMARK ĐỐI CHIẾU LLM PARSER & ABLATION STUDY (RQ3)
    # Mặc định gọi LLM THẬT (cần GOOGLE_API_KEY trong .env). Dùng --offline để ép
    # dùng bộ giả lập deterministic (chỉ cho CI smoke-test, KHÔNG dùng để công bố).
    print("\n[BƯỚC 7/7] Đang chạy benchmark đối chiếu LLM Parser (Ablation Study 50 prompts)...")
    run_parser_benchmark()

    print("\n" + "=" * 95)
    print(" 🎉 HOÀN THÀNH TOÀN BỘ SUITE KIỂM THỬ TỔNG HỢP!")
    print(" 📊 Kết quả đã được quy tụ gọn gàng về thư mục: data/processed/scm_results/")
    print("    - test_f1_rmse_evaluation.csv         (Accuracy SCM: bucket + full-resolution)")
    print("    - 05_model_comparison.csv             (Đối chiếu 4 mô hình: bucket + full-resolution)")
    print("    - p_value_statistical_test.csv        (Wilcoxon/Friedman theo từng metric, dùng mape_pct)")
    print("    - ground_truth_direct_match.csv       (407k+ điểm so khớp thật, không bucket)")
    print("    - ground_truth_direct_match_summary.csv (Tổng hợp theo service/metric)")
    print("    - parser_ablation_benchmark.csv       (RQ3: LLM thật, N lần lặp, mean±std)")
    print("    - parser_ablation_by_category.csv     (RQ3: breakdown theo 4 nhóm prompt)")
    print("    - docs/table_rq3_parser_ablation.tex  (Bảng LaTeX)")
    print("    - docs/RQ3_LLM_PARSER_BENCHMARK_REPORT.md (Báo cáo RQ3, sinh tự động từ CSV)")
    print("=" * 95)


if __name__ == '__main__':
    run_full_suite()
