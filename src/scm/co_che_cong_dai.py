# -*- coding: utf-8 -*-
"""CO CHE CO CONG THEO DAI cho node co cha la TAI NGUYEN (canh Tier-2.5 `R -> R`).

VI SAO TON TAI
==============
Canh backpressure `caller_cpu -> callee_cpu` duoc chung nhan bang giao thuc
held-out OOD: cai thien o 23/23 node (3/3 Sock Shop, 20/20 Train Ticket), va
giam so ca doi dau. Tren bang chung do no da duoc trien khai.

Phep kiem CAN THIEP bac bo no. Do tren 1080 quan sat fault-injection
(`experiments/chua_phan_loai/rq7_interventional_validity.py`):

    suy giam MAPE trung binh duoi can thiep
      co che hai tang  (R_B ~ W_B)   :    12,6
      canh R->R        (R_B ~ R_A)   : 1 658,3      <- te hon 132 lan
      xau nhat                        : 438 101

Nguyen nhan da dinh vi: cha la TAI NGUYEN ra ngoai dai gia tri da thay luc fit
o 74,7% hang duoi can thiep, nhung chi 0,1% o du lieu held-out -- nen giao thuc
held-out GAN NHU KHONG BAO GIO tao ra dieu kien lam tuyen bo sai. Trong dai thi
hai dang khong phan biet duoc (ti le suy giam trung vi 1,00x).

PHEP VA -- KEP, KHONG chuyen nhanh. Ra ngoai dai thi KEP gia tri cha tai nguyen
ve bien dai (`clip` vao [lo, hi]) roi van dung mo hinh day.

Vi sao KHONG chuyen nhanh sang "chi dung cha workload": ban dau toi cai kieu
chuyen nhanh va no PHA TINH DON DIEU -- do duoc: voi cha workload giu co dinh,
du bao TUT 7,09 ngay tai bien dai khi cha tai nguyen tang qua `hi`. Do la vi pham
tien dieu kien (a) cua Menh de 2 (Certified capacity envelope), va te hon la
`_check_monotone_precondition` trong capacity_agent.py KHONG bat duoc: no chi xet
`coef_ >= 0`, ma `coef_` cua nhanh day van khong am.

KEP thi khong co van de do:
  * r <= hi : dung mo hinh day, don dieu theo moi bien (moi he so khong am)
  * r >  hi : hang so theo r (khong giam), van don dieu theo workload
  * lien tuc tai bien, khong co cu nhay
va no van chan dung thu can chan: mo hinh NGOAI SUY TUYEN TINH tren gia tri cha
tai nguyen chua tung thay -- dung co che that bai da do.

KET QUA: xem bang trong phan "do lai sau khi doi sang KEP" -- cac so cua ban
CHUYEN NHANH khong con ap dung.

THU TU COT: `dowhy.graph.get_ordered_predecessors` tra ve `sorted(predecessors)`,
nen chi so cot phu thuoc TEN node va phai tinh theo TUNG node -- xem
`chi_so_cha_workload()`.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


def chi_so_cha_workload(g, node):
    """Chi so cac cot `_workload` trong X cua `node`, theo DUNG thu tu cua dowhy.

    Tra ve (idx_workload, so_cha). Goi ben ngoai de biet node co can cong khong:
    chi can cong khi node co CA cha workload LAN cha tai nguyen, tuc
    `0 < len(idx) < so_cha`.
    """
    cha = sorted(g.predecessors(node))
    return tuple(i for i, c in enumerate(cha) if c.endswith('_workload')), len(cha)


class CoCheCongDai(BaseEstimator, RegressorMixin):
    """Hoi quy khong am, co cong theo dai tren cac cha KHONG phai workload.

    `idx_an_toan`: chi so cac cot duoc coi la an toan (cha workload) -- lay tu
    `chi_so_cha_workload()`. Cac cot con lai la cha tai nguyen, bi dat cong.
    """

    def __init__(self, idx_an_toan=(0,)):
        self.idx_an_toan = tuple(idx_an_toan)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if not self.idx_an_toan or len(self.idx_an_toan) >= X.shape[1]:
            raise ValueError('CoCheCongDai can CA cha workload LAN cha tai nguyen; '
                             f'idx_an_toan={self.idx_an_toan}, so cot={X.shape[1]}')
        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        self.idx_cong_ = tuple(i for i in range(X.shape[1]) if i not in self.idx_an_toan)
        self.m_day_ = LinearRegression(positive=True).fit(X, y)
        # `.coef_` / `.intercept_` gia lap theo nhanh DAY, de moi code doc truc tiep
        # thuoc tinh kieu sklearn van chay: `_check_monotone_precondition` trong
        # capacity_agent.py doc `.coef_`, con notebook 01 muc A4 doc CA `.intercept_`.
        # Ca hai mo hinh con deu khong am nen tinh don dieu -- tien dieu kien cua
        # Menh de 2 -- duoc giu o ca hai nhanh cua cong.
        self.coef_ = self.m_day_.coef_
        self.intercept_ = self.m_day_.intercept_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).copy()
        # KEP cha tai nguyen ve bien dai da thay luc fit. Cac cot an toan
        # (workload) KHONG bi kep -- chung duoc phep ngoai suy, va do chinh la
        # thuoc tinh phan biet co che hai tang: cha workload ra ngoai dai o
        # 0,1% hang (held-out) va it hon han cha tai nguyen duoi can thiep.
        for i in self.idx_cong_:
            X[:, i] = np.clip(X[:, i], self.lo_[i], self.hi_[i])
        return self.m_day_.predict(X)

    def ty_le_trong_dai(self, X):
        """Ti le hang KHONG bi kep (cha tai nguyen con trong dai) -- de bao cao."""
        X = np.asarray(X, dtype=float)
        trong = np.ones(len(X), dtype=bool)
        for i in self.idx_cong_:
            trong &= (X[:, i] >= self.lo_[i]) & (X[:, i] <= self.hi_[i])
        return float(trong.mean())
