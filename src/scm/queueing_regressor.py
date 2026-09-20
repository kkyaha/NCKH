# -*- coding: utf-8 -*-
"""
QUEUEING LATENCY REGRESSOR (M/M/1-style, dung chung cho Global DAG)
======================================================================
Tach ra tu capacity_agent.py de deterministic_forward.py (src/scm/, KHONG
duoc phu thuoc nguoc vao src/agents/ theo dung phan tang cua repo -- xem
README "Vi sao tach vay") co the isinstance-check dung class nay ma khong
tao vong lap import. capacity_agent.py import lai class nay tu day (giu
nguyen ten, hanh vi khong doi).

LUU Y: cac ban sao rieng trong experiments/evaluation_suite.py,
experiments/select_scm_edges.py, experiments/backpressure_edge_ood_safety_test.py
la co y -- do la ha tang thuc nghiem sinh so lieu cho paper (RQ1/RQ4, cac
guard test), KHONG duoc doi de tranh lam lech so lieu da cong bo (xem README
"Vi sao tach vay"). Module nay CHI la nguon that cho code san pham
(src/agents/capacity_agent.py va cac module src/scm/ khac).

CANH BAO da kiem chung (README "Gioi han da biet" #7): tren Train Ticket,
CA 28/28 co che nay fit ra coef_ == [0, 0] (khong chi 1/7 node nhu Sock
Shop tung ghi nhan o tren) -- tuong quan workload<->latency tho gan nhu
khong co (~0.0016) trong du lieu "binh thuong" cua Train Ticket. Dung
node_impact.evaluate_node_stability() de kiem lai bat ky he thong moi nao
truoc khi tin so lieu latency cua no.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class QueueingLatencyRegressor(BaseEstimator, RegressorMixin):
    """
    Mo hinh hang doi phi tuyen (M/M/1 - Kleinrock approximation):
    Khi Workload -> Capacity, Latency bung no theo ham tiem can: W / (C - W).

    positive=True (fix): ban dau model_ dung LinearRegression() KHONG rang
    buoc dau. Vi W tho va W/(C-W) tuong quan cao (gan collinear), hoi quy
    khong rang buoc doi khi gan he so AM cho W tho de bu tru -- pha vo tinh
    don dieu ma Proposition 2 (paper, "Certified Capacity Envelope") can:
    kiem tren toan bo 7 mechanism latency cua Sock Shop, ca 7/7 co he so am
    (vi du orders_latency-50 du bao Y(delta=5%) > Y(delta=50%), tuc tai
    tang ma latency du bao GIAM). Bat positive=True khong chi phuc hoi tinh
    don dieu (6/7 sach ngay, node con lai chi con nhieu Monte Carlo tren he
    so ~0, khong phai vi pham cau truc) ma con CAI THIEN MAPE held-out
    (protocol OOD 67/33 giong Section "Experimental Setup") tren ca 7/7
    dich vu -- he so am von la dau hieu overfit do collinearity, khong phai
    tin hieu that. Khong phai danh doi, ma la sua mot loi fit khong on dinh.
    """
    def __init__(self):
        self.model_ = LinearRegression(fit_intercept=True, positive=True)
        self.capacity_ = None

    def fit(self, X, y):
        X = np.array(X)
        # P99 thay vi max(X): thong ke on dinh theo co mau -- hoi tu ve
        # phan vi that cua phan phoi khi them du lieu (LLN), khac max, mot
        # thong ke cuc tri tang khong gioi han theo co mau tren duoi dai.
        # Da xac nhan la nguyen nhan that: mo rong cua so Alibaba tu 5h len
        # 10h lam max(X) tang, day capacity_ len theo, xoa mat tin hieu phi
        # tuyen va lam bang chung Proposition 3 tren du lieu that sup ve 0
        # (xem PROGRESS_REPORT muc 3.3).
        self.capacity_ = np.percentile(X, 99, axis=0) * 1.5
        self.capacity_[self.capacity_ == 0] = 1.0
        # Khac voi truoc day (max(X)*1.5 dam bao > moi diem X), P99*1.5
        # KHONG con chac chan vuot moi diem trong duoi dai -- chan X truoc
        # khi tinh X_queue, giong het predict(), de tranh mau so am/no.
        X_capped = np.minimum(X, self.capacity_ * 0.99)
        X_queue = X_capped / (self.capacity_ - X_capped + 1e-6)
        X_transformed = np.hstack([X_capped, X_queue])
        self.model_ = LinearRegression(fit_intercept=True, positive=True)
        self.model_.fit(X_transformed, y)
        return self

    def predict(self, X):
        X = np.array(X)
        X_capped = np.minimum(X, self.capacity_ * 0.99)
        X_queue = X_capped / (self.capacity_ - X_capped + 1e-6)
        return self.model_.predict(np.hstack([X_capped, X_queue]))


__all__ = ['QueueingLatencyRegressor']
