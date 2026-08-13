# -*- coding: utf-8 -*-
"""
KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (STATISTICAL SIGNIFICANCE TESTING FOR Q1 PAPERS)
===========================================================================
Thực hiện kiểm định Wilcoxon Signed-Rank Test & Friedman Test đối chiếu
sai số RMSE / F1 giữa SCM và các mô hình Baseline (LinearReg, GradBoost, GaussProc).
Xuất kết quả p-value chuẩn báo chí Q1 IEEE/ACM.
"""

import os
import sys
import warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r'c:\NGUYEN KHANH KY\NCKH\mas_architecture_project'
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', '05_model_comparison.csv')
OUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', 'p_value_statistical_test.csv')


def run_statistical_significance():
    print("=" * 90)
    print("  CHẠY KIỂM ĐỊNH THỐNG KÊ (WILCOXON & FRIEDMAN TESTS) CHO BÀI BÁO Q1")
    print("=" * 90)

    if not os.path.exists(CSV_PATH):
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}. Hãy chạy model_comparison.py trước.")
        return

    df = pd.read_csv(CSV_PATH)
    valid = df.dropna(subset=['rmse', 'mape_pct', 'f1_score'])

    scm_rmse = valid[valid['model'] == 'SCM_DoWhy']['rmse'].values
    lr_rmse  = valid[valid['model'] == 'LinearReg']['rmse'].values
    gb_rmse  = valid[valid['model'] == 'GradBoost']['rmse'].values
    gp_rmse  = valid[valid['model'] == 'GaussianProcess']['rmse'].values

    # Pairwise Wilcoxon Signed-Rank Tests
    w_lr_stat, w_lr_p = stats.wilcoxon(scm_rmse, lr_rmse) if len(scm_rmse) == len(lr_rmse) else (0, 1)
    w_gb_stat, w_gb_p = stats.wilcoxon(scm_rmse, gb_rmse) if len(scm_rmse) == len(gb_rmse) else (0, 1)
    w_gp_stat, w_gp_p = stats.wilcoxon(scm_rmse, gp_rmse) if len(scm_rmse) == len(gp_rmse) else (0, 1)

    # Overall Friedman Test across all 4 models
    min_len = min(len(scm_rmse), len(lr_rmse), len(gb_rmse), len(gp_rmse))
    if min_len > 0:
        f_stat, f_p = stats.friedmanchisquare(scm_rmse[:min_len], lr_rmse[:min_len], gb_rmse[:min_len], gp_rmse[:min_len])
    else:
        f_stat, f_p = 0, 1

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

    print("\n" + "=" * 90)
    print(f"  ✅ ĐÃ XUẤT BẢNG KIỂM ĐỊNH THỐNG KÊ CHO BÀI BÁO Q1 TẠI: {OUT_PATH}")
    print("=" * 90)

if __name__ == '__main__':
    run_statistical_significance()
