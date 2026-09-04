# -*- coding: utf-8 -*-
"""
PerformanceAgent - SCM-Based Performance Prediction
====================================================
Agent đánh giá hiệu năng hệ thống dựa trên mô hình nhân quả cấu trúc (SCM)
đã được train và test hoàn chỉnh theo phương pháp Gold Standard:
  - Train: dữ liệu bình thường tại mức tải THẤP (bottom 67% workload)
  - Test : dự đoán tại mức tải CAO (top 33%) và so sánh với dữ liệu thật

Thay thế hoàn toàn kết nối BigQuery bằng SCM model được huấn luyện cục bộ.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
# Tự động resolve BASE_DIR từ vị trí file này (src/agents/) -> project root
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_AGENT_DIR)
BASE_DIR   = os.path.dirname(_SRC_DIR)

# Thêm src/scm vào path để import data_processor (nếu cần)
sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))

# ============================================================
# CONSTANTS (đồng bộ với scm_pipeline.py)
# ============================================================
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']

METRICS = [
    ('CPU',    'cpu',    '%',   1.0   ),
    ('Memory', 'mem',    'MB',  1/1e6 ),
    ('Socket', 'socket', 'cnt', 1.0   ),
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

N_PROJ = 500  # Số mẫu Monte Carlo cho interventional_samples


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def _load_normal_data(service: str, metric_col: str, data_dir: str) -> pd.DataFrame:
    """Load normal-period [Workload, Target] từ tất cả các run."""
    dfs = []
    wlc = f'{service}_workload'
    tgc = f'{service}_{metric_col}'

    target_dir = os.path.join(data_dir, 'RE2-SS') if os.path.isdir(os.path.join(data_dir, 'RE2-SS')) else data_dir

    for scenario in os.listdir(target_dir):
        sp = os.path.join(target_dir, scenario)
        if not os.path.isdir(sp):
            continue
        for run_id in os.listdir(sp):
            rp = os.path.join(sp, run_id)
            if not os.path.isdir(rp):
                continue
            mp = os.path.join(rp, 'simple_metrics.csv')
            ip = os.path.join(rp, 'inject_time.txt')
            if not (os.path.exists(mp) and os.path.exists(ip)):
                continue
            try:
                with open(ip) as f:
                    it = int(f.read().strip())
                cols = pd.read_csv(mp, nrows=0).columns
                tc = 'imte' if 'imte' in cols else ('time' if 'time' in cols else None)
                if tc is None or wlc not in cols or tgc not in cols:
                    continue
                df_raw = pd.read_csv(mp, usecols=[tc, wlc, tgc])
                df = df_raw[[tc, wlc, tgc]].copy()
                df.columns = [tc, 'Workload', 'Target']
                dfs.append(df[df[tc] < it].drop(columns=[tc]).dropna())
            except Exception:
                continue

    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================
# PERFORMANCE AGENT
# ============================================================
class PerformanceAgent:
    """
    Agent đánh giá hiệu năng vi dịch vụ dựa trên mô hình SCM.

    Workflow:
      1. train_scm_models()             - Huấn luyện SCM cho từng (service, metric)
      2. get_metrics_for_service()      - Truy vấn dự đoán theo workload (API cũ)
      3. predict_workload_impact()      - Dự đoán tác động tại workload tùy ý
      4. get_system_impact_for_request()- Đánh giá toàn hệ thống theo request type
      5. get_accuracy_report()          - Báo cáo độ chính xác (MAPE, F1, R2...)
      6. get_workload_sensitivity()     - Nhạy cảm theo mức tăng workload
    """

    def __init__(self, data_dir: str = None, auto_train: bool = True):
        """
        Khởi tạo PerformanceAgent.

        Args:
            data_dir: Đường dẫn đến thư mục data/raw.
                      Nếu None, tự động tìm từ vị trí file này.
            auto_train: Nếu True, tự động huấn luyện SCM ngay khi khởi tạo.
        """
        if data_dir is None:
            data_dir = os.path.join(BASE_DIR, 'data', 'raw')

        self.data_dir = data_dir
        self.trained_models: dict = {}       # key: (service, metric_name)
        self.accuracy_df: pd.DataFrame = None
        self._is_trained = False

        print(f"[PerformanceAgent] Data directory: {self.data_dir}")

        if auto_train:
            self.train_scm_models()

    # ----------------------------------------------------------
    # BƯỚC 1: HUẤN LUYỆN MÔ HÌNH SCM (GOLD STANDARD PIPELINE)
    # ----------------------------------------------------------
    def train_scm_models(self):
        """
        Huấn luyện SCM bivariate (Workload -> Target) cho từng (service, metric).
        Phương pháp Gold Standard:
          - Train: bottom 67% workload (tải thấp - bình thường)
          - Test : top 33%  workload (tải cao  - out-of-distribution)
        Lưu kết quả vào self.trained_models và self.accuracy_df.
        """
        print("\n" + "=" * 70)
        print("  [PerformanceAgent] HUAN LUYEN MO HINH SCM")
        print("  Phuong phap: Train(LOW workload 67%) -> Test(HIGH workload 33%)")
        print("=" * 70)

        if not os.path.isdir(self.data_dir):
            print(f"[WARNING] Thu muc du lieu khong ton tai: {self.data_dir}")
            print("  PerformanceAgent se hoat dong o che do khong co du lieu.")
            self._is_trained = False
            return

        try:
            from data_processor import load_multi_service_data
            df_multi = load_multi_service_data(self.data_dir)
        except Exception:
            df_multi = None

        all_rows = []

        for metric_name, metric_col, unit, scale in METRICS:
            print(f"\n  [{metric_name} | don vi: {unit}]")
            for svc in SERVICES:
                wlc, tgc = f'{svc}_workload', f'{svc}_{metric_col}'
                if df_multi is not None and wlc in df_multi.columns and tgc in df_multi.columns:
                    df = df_multi[[wlc, tgc]].dropna()
                    df.columns = ['Workload', 'Target']
                else:
                    df = _load_normal_data(svc, metric_col, self.data_dir)

                if df is None or len(df) < 200:
                    print(f"    {svc:<14}: khong du du lieu (can >=200 mau)")
                    continue

                # Sample toi da 2,000 mau de dong bo voi SimulationAgent va toi uu thoi gian train
                if len(df) > 2000:
                    df = df.sample(2000, random_state=42)

                # Chia train/test theo workload (Gold Standard)
                df = df.sort_values('Workload').reset_index(drop=True)
                split = int(len(df) * 0.67)
                df_train = df.iloc[:split]
                df_test  = df.iloc[split:].copy()

                # Xây dựng & huấn luyện SCM: Workload -> Target
                g = nx.DiGraph()
                g.add_edge('Workload', 'Target')
                model = gcm.InvertibleStructuralCausalModel(g)
                gcm.auto.assign_causal_mechanisms(model, df_train)
                gcm.fit(model, df_train)

                # Đánh giá trên tập Test (bucket by workload)
                df_test['bkt'] = pd.qcut(
                    df_test['Workload'],
                    q=min(8, df_test['Workload'].nunique()),
                    duplicates='drop'
                )
                bkts = df_test.groupby('bkt', observed=True)[['Workload', 'Target']].mean()

                y_true, y_pred = [], []
                for _, row in bkts.iterrows():
                    wl_val = row['Workload']
                    dp = gcm.interventional_samples(
                        model,
                        interventions={'Workload': lambda x, w=wl_val: w},
                        num_samples_to_draw=N_PROJ
                    )
                    y_true.append(row['Target'])
                    y_pred.append(dp['Target'].mean())

                yt, yp = np.array(y_true), np.array(y_pred)
                mape_v = _mape(yt, yp)
                mae_v  = mean_absolute_error(yt, yp) * scale
                rmse_v = np.sqrt(mean_squared_error(yt, yp)) * scale
                r2_v   = r2_score(yt, yp)

                # Risk Detection F1-Score (Threshold = Mean_train + 0.5*Std_train)
                thresh = df_train['Target'].mean() + 0.5 * df_train['Target'].std()
                yt_bin = (yt >= thresh).astype(int)
                yp_bin = (yp >= thresh).astype(int)
                f1_v   = f1_score(yt_bin, yp_bin, average='binary', zero_division=1)

                tag = 'EXCELLENT' if mape_v < 10 else ('FAIR' if mape_v < 25 else 'POOR')
                print(f"    {svc:<14}: MAPE={mape_v:>5.1f}%  "
                      f"RMSE={rmse_v:>8.4f}  F1={f1_v:>5.3f}  R2={r2_v:>5.3f}  [{tag}]")

                # Lưu model và metadata
                self.trained_models[(svc, metric_name)] = {
                    'model':        model,
                    'baseline_wl':  df_train['Workload'].mean(),
                    'baseline_val': df_train['Target'].mean() * scale,
                    'wl_train_max': df_train['Workload'].max(),
                    'wl_test_max':  df_test['Workload'].max(),
                    'unit':         unit,
                    'scale':        scale,
                    'mape':         mape_v,
                    'f1':           f1_v,
                    'r2':           r2_v,
                    'rmse':         rmse_v,
                    'thresh':       thresh,
                }

                all_rows.append({
                    'service':   svc,
                    'metric':    metric_name,
                    'unit':      unit,
                    'n_train':   len(df_train),
                    'n_test':    len(df_test),
                    'mape_pct':  round(mape_v, 2),
                    'mae':       round(mae_v,  4),
                    'rmse':      round(rmse_v, 4),
                    'f1_score':  round(f1_v,   3),
                    'r2':        round(r2_v,   3),
                })

        self.accuracy_df = pd.DataFrame(all_rows)
        self._is_trained = len(self.trained_models) > 0

        # Tóm tắt
        print("\n  --- Tom tat theo Metric ---")
        for metric_name, _, _, _ in METRICS:
            if self.accuracy_df.empty:
                continue
            sub = self.accuracy_df[self.accuracy_df['metric'] == metric_name]
            if sub.empty:
                continue
            n_good = (sub['mape_pct'] < 10).sum()
            print(f"  {metric_name:<14}: MAPE trung binh={sub['mape_pct'].mean():>5.1f}% | "
                  f"{n_good}/{len(sub)} dich vu EXCELLENT (<10%)")

        n_models = len(self.trained_models)
        print(f"\n[OK] Da huan luyen xong {n_models} mo hinh SCM "
              f"({len(SERVICES)} dich vu x {len(METRICS)} metrics)\n")

    # ----------------------------------------------------------
    # BƯỚC 2: DỰ ĐOÁN CHO 1 DỊCH VỤ (tương thích API cũ)
    # ----------------------------------------------------------
    def get_metrics_for_service(
        self,
        service_name: str,
        workload_delta_pct: float = 20.0,
        scenario: str = None   # kept for backward compatibility, not used
    ) -> dict:
        """
        Lấy metrics dự đoán cho 1 dịch vụ tại mức workload tăng thêm X%.
        Tương thích với API cũ (thay thế BigQuery query).

        Args:
            service_name: Tên dịch vụ (vd: 'front-end', 'payment').
            workload_delta_pct: Phần trăm tăng workload so với baseline (mặc định 20%).
            scenario: (bỏ qua - giữ để tương thích API cũ).

        Returns:
            dict: Các keys: '{metric}_baseline_{unit}', '{metric}_predicted_{unit}',
                            '{metric}_change_pct'
        """
        if not self._is_trained:
            print(f"[WARNING] SCM chua duoc huan luyen. Goi train_scm_models() truoc.")
            return {}

        result = {}
        for metric_name, _, unit, scale in METRICS:
            key = (service_name, metric_name)
            if key not in self.trained_models:
                continue

            m_info  = self.trained_models[key]
            model   = m_info['model']
            base_wl = m_info['baseline_wl']
            base_v  = m_info['baseline_val']
            new_wl  = base_wl * (1 + workload_delta_pct / 100)

            dp = gcm.interventional_samples(
                model,
                interventions={'Workload': lambda x, w=new_wl: w},
                num_samples_to_draw=N_PROJ
            )
            pred_v = dp['Target'].mean() * scale
            chg    = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0

            result[f'{metric_name}_baseline_{unit}']  = round(base_v, 4)
            result[f'{metric_name}_predicted_{unit}'] = round(pred_v, 4)
            result[f'{metric_name}_change_pct']       = round(chg, 2)

        return result

    # ----------------------------------------------------------
    # BƯỚC 3: DỰ ĐOÁN TẠI WORKLOAD TÙY Ý
    # ----------------------------------------------------------
    def predict_workload_impact(self, service_name: str, new_workload: float) -> dict:
        """
        Dự đoán tài nguyên tiêu thụ tại một giá trị workload cụ thể (req/s).

        Returns:
            dict: {metric_name: {'unit', 'baseline', 'predicted', 'change_pct'}}
        """
        if not self._is_trained:
            print(f"[WARNING] SCM chua duoc huan luyen.")
            return {}

        result = {}
        for metric_name, _, unit, scale in METRICS:
            key = (service_name, metric_name)
            if key not in self.trained_models:
                continue

            m_info = self.trained_models[key]
            model  = m_info['model']
            base_v = m_info['baseline_val']

            dp = gcm.interventional_samples(
                model,
                interventions={'Workload': lambda x, w=new_workload: w},
                num_samples_to_draw=N_PROJ
            )
            pred_v = dp['Target'].mean() * scale
            chg    = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0

            result[metric_name] = {
                'unit':       unit,
                'baseline':   round(base_v, 4),
                'predicted':  round(pred_v, 4),
                'change_pct': round(chg, 2),
            }

        return result

    # ----------------------------------------------------------
    # BƯỚC 4: TÁC ĐỘNG TOÀN HỆ THỐNG THEO LOẠI REQUEST
    # ----------------------------------------------------------
    def get_system_impact_for_request(
        self,
        request_type: str,
        workload_delta_pct: float = 20.0
    ) -> dict:
        """
        Đánh giá tác động toàn hệ thống khi thêm 1 loại request mới.

        Args:
            request_type: Key trong CALL_CHAINS (vd: 'APPLY_PROMO_CODE').
            workload_delta_pct: % tăng workload trên mỗi dịch vụ bị ảnh hưởng.

        Returns:
            dict: {'request_type', 'blast_radius', 'workload_delta_pct', 'impact'}
        """
        if not self._is_trained:
            print(f"[WARNING] SCM chua duoc huan luyen.")
            return {}

        if request_type in CALL_CHAINS:
            affected_services = CALL_CHAINS[request_type]
        else:
            affected_services = list({s for (s, _) in self.trained_models.keys()})

        impact = {}
        for svc in affected_services:
            metrics = self.get_metrics_for_service(svc, workload_delta_pct)
            if metrics:
                impact[svc] = metrics

        return {
            'request_type':       request_type,
            'blast_radius':       affected_services,
            'workload_delta_pct': workload_delta_pct,
            'impact':             impact,
        }

    # ----------------------------------------------------------
    # BƯỚC 5: BÁO CÁO ĐỘ CHÍNH XÁC MÔ HÌNH
    # ----------------------------------------------------------
    def get_accuracy_report(self) -> pd.DataFrame:
        """
        Trả về DataFrame MAPE/MAE/RMSE/F1/R2 của từng (service, metric).
        Hữu ích để LLM đánh giá độ tin cậy của dự đoán.
        """
        if self.accuracy_df is None or self.accuracy_df.empty:
            return pd.DataFrame()
        return self.accuracy_df.copy()

    def get_accuracy_summary(self) -> dict:
        """
        Tóm tắt độ chính xác dưới dạng dict (dễ nhúng vào LLM prompt).
        """
        if self.accuracy_df is None or self.accuracy_df.empty:
            return {}

        summary = {}
        for metric_name, _, _, _ in METRICS:
            sub = self.accuracy_df[self.accuracy_df['metric'] == metric_name]
            if sub.empty:
                continue
            n_good = int((sub['mape_pct'] < 10).sum())
            summary[metric_name] = {
                'avg_mape_pct':       round(float(sub['mape_pct'].mean()), 2),
                'avg_f1':             round(float(sub['f1_score'].mean()), 3),
                'avg_r2':             round(float(sub['r2'].mean()), 3),
                'excellent_services': n_good,
                'total_services':     len(sub),
            }
        return summary

    # ----------------------------------------------------------
    # BƯỚC 6: WORKLOAD SENSITIVITY (tương đương STEP 3 của scm_pipeline)
    # ----------------------------------------------------------
    def get_workload_sensitivity(
        self,
        service_name: str,
        pct_list: list = None
    ) -> pd.DataFrame:
        """
        Dự đoán thay đổi tài nguyên khi workload tăng +10%, +20%, +30%, +50%.

        Returns:
            DataFrame: Bảng dự đoán theo từng mức tăng workload.
        """
        if pct_list is None:
            pct_list = [10, 20, 30, 50]

        if not self._is_trained:
            return pd.DataFrame()

        rows = []
        for metric_name, _, unit, _ in METRICS:
            key = (service_name, metric_name)
            if key not in self.trained_models:
                continue

            m_info  = self.trained_models[key]
            base_wl = m_info['baseline_wl']
            base_v  = m_info['baseline_val']
            model   = m_info['model']
            scale   = m_info['scale']

            for pct in pct_list:
                new_wl = base_wl * (1 + pct / 100)
                dp = gcm.interventional_samples(
                    model,
                    interventions={'Workload': lambda x, w=new_wl: w},
                    num_samples_to_draw=N_PROJ
                )
                pred_v = dp['Target'].mean() * scale
                chg    = (pred_v - base_v) / abs(base_v) * 100 if base_v != 0 else 0.0

                rows.append({
                    'service':               service_name,
                    'metric':                metric_name,
                    'unit':                  unit,
                    'baseline_workload':     round(base_wl, 2),
                    'workload_increase_pct': pct,
                    'baseline_value':        round(base_v, 4),
                    'predicted_value':       round(pred_v, 4),
                    'change_pct':            round(chg, 2),
                })

        return pd.DataFrame(rows)
