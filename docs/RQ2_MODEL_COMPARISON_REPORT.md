# RQ2 — SCM (nhân quả) so với các mô hình tương quan/hộp đen trong vùng OOD

> **RQ2:** Mô hình nhân quả (SCM + do-calculus) có vượt trội các mô hình tương quan/hộp đen
> truyền thống (Linear Regression, Gradient Boosting, Gaussian Process) khi dự báo ngoài
> phân phối (OOD) hay không?

Sinh tự động từ `05_model_comparison.csv` và `p_value_statistical_test.csv`.

---

## 1. Thiết lập
Cùng giao thức OOD Gold Standard như RQ1 (train tải thấp → test tải cao), áp dụng đồng thời cho
4 mô hình trên cùng dữ liệu. Báo cáo trên **MAPE full-resolution** (toàn bộ điểm test thô, không
bucket-average) — chặt chẽ hơn bản bucket-average dùng trong các bản nháp trước.

## 2. MAPE trung bình theo Model × Metric (full-resolution)

| Metric | LinearReg | GradBoost | GaussianProcess | SCM_DoWhy |
|---|---|---|---|---|
| CPU | 46.76% | 35.93% | 38.37% | 42.54% |
| Memory | 3.94% | 4.63% | 3.70% | 3.99% |
| Socket | 20.83% | 15.60% | 19.27% | 14.57% |

## 3. Số lần thắng theo (service, metric) — 21 tổ hợp

| Model | Số lần thắng (thấp nhất MAPE full-res) / 21 |
|---|---|
| LinearReg | 1 |
| GradBoost | 6 |
| GaussianProcess | 7 |
| SCM_DoWhy | 7 |

## 4. Kiểm định ý nghĩa thống kê (Wilcoxon/Friedman, tách riêng từng metric — KHÔNG gộp đơn vị RMSE khác nhau)

| So sánh | Metric | n | Median Δ (SCM−baseline) | Wilcoxon stat | p-value | Ý nghĩa |
|---|---|---|---|---|---|---|
| SCM_vs_LinearReg | CPU | 7 | -4.99 | 3.0 | 0.07812 | ❌ Không |
| SCM_vs_GradBoost | CPU | 7 | 0.86 | 7.0 | 0.29688 | ❌ Không |
| SCM_vs_GaussianProcess | CPU | 7 | 0.12 | 13.0 | 0.9375 | ❌ Không |
| Friedman_4_Models | CPU | 7 | nan | 4.543 | 0.2085 | ❌ Không |
| SCM_vs_LinearReg | Memory | 7 | 0.03 | 7.0 | 0.46307 | ❌ Không |
| SCM_vs_GradBoost | Memory | 7 | -0.01 | 9.0 | 0.46875 | ❌ Không |
| SCM_vs_GaussianProcess | Memory | 7 | 0.18 | 1.0 | 0.07962 | ❌ Không |
| Friedman_4_Models | Memory | 7 | nan | 6.231 | 0.10091 | ❌ Không |
| SCM_vs_LinearReg | Socket | 7 | -1.31 | 1.0 | 0.03125 | ✅ Có |
| SCM_vs_GradBoost | Socket | 7 | -0.03 | 7.0 | 0.29688 | ❌ Không |
| SCM_vs_GaussianProcess | Socket | 7 | -1.03 | 3.0 | 0.07812 | ❌ Không |
| Friedman_4_Models | Socket | 7 | nan | 7.174 | 0.06656 | ❌ Không |
| SCM_vs_LinearReg | ALL_COMBINED | 21 | -0.94 | 39.0 | 0.01374 | ✅ Có |
| SCM_vs_GradBoost | ALL_COMBINED | 21 | -0.01 | 108.0 | 0.81168 | ❌ Không |
| SCM_vs_GaussianProcess | ALL_COMBINED | 21 | 0.0 | 75.0 | 0.42091 | ❌ Không |
| Friedman_4_Models | ALL_COMBINED | 21 | nan | 5.221 | 0.15634 | ❌ Không |

**Cách đọc**: `median_diff_SCM_minus_baseline` **âm** nghĩa là SCM có MAPE **thấp hơn** (tốt hơn)
baseline đó.

---

## 5. Kết luận trung thực (không phóng đại)
* SCM **có ý nghĩa thống kê** so với LinearRegression khi gộp toàn bộ (p=0.01374, median Δ=-0.94 → MAPE của SCM thấp hơn (tốt hơn)).
* SCM **không khác biệt có ý nghĩa** so với GradBoost hay GaussianProcess (p>0.05 ở mọi phép so sánh gộp toàn bộ) — tức là SCM đạt độ chính xác **tương đương**, không vượt trội, so với 2 baseline ML phức tạp hơn.
* p>0.05 **không chứng minh** hai mô hình bằng nhau (chỉ là không đủ bằng chứng bác bỏ); muốn
  khẳng định "tương đương" chặt chẽ cần kiểm định equivalence (TOST), chưa thực hiện.
* **Giá trị thực sự của SCM không nằm ở độ chính xác điểm** (ngang bằng hoặc chỉ nhỉnh hơn đôi
  chút baseline), mà ở khả năng **can thiệp do(x)** — thứ GradBoost/GaussianProcess/LinearRegression
  không hỗ trợ. Đây nên là luận điểm chính khi viết RQ2, không phải "SCM chính xác hơn".

File nguồn: `05_model_comparison.csv`, `p_value_statistical_test.csv`.
