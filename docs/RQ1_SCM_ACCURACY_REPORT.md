# RQ1 — Độ chính xác dự báo OOD của SCM (do-calculus)

> ⚠️ **TÀI LIỆU THEO HỆ ĐÁNH SỐ RQ CŨ (RQ1–RQ12).** Số RQ trong file này **không khớp**
> khung 5 RQ hiện tại của `docs/paper_draft.tex`. Xem bảng dịch ở đầu `README.md` gốc.
> Giữ lại làm hồ sơ gốc; **không dùng làm nguồn số liệu**.

> **RQ1:** SCM kết hợp do-calculus dự đoán tài nguyên (CPU/Memory/Socket) chính xác đến mức nào
> khi ngoại suy sang vùng tải cao chưa từng quan sát (Out-of-Distribution)?

Tài liệu này sinh **tự động** từ `test_f1_rmse_evaluation.csv` và
`ground_truth_direct_match_summary.csv` — không gõ tay số liệu.

> **Cập nhật:** cơ chế hồi quy Bivariate dùng để sinh 2 file CSV trên đã đổi từ
> `gcm.auto.assign_causal_mechanisms` (tự chọn mô hình) sang
> `LinearRegression(positive=True)` tường minh — đồng bộ với cơ chế Global DAG và
> Fast Path production (`capacity_agent.py`), đúng claim "uniformly" của paper ở
> Section "Extrapolation-Sign Failure Mode". Số liệu dưới đây đã được **chạy lại**
> với cơ chế mới; nhìn chung MAPE full-resolution CPU/Socket **tăng** so với bản
> trước (đổi lại là đảm bảo không còn sign-inversion khi ngoại suy) — xem
> `experiments/nonlinear_mechanism_trial.py` và `docs/paper_draft.tex` mục
> "Extrapolation-Sign Failure Mode" để biết bối cảnh đầy đủ.

---

## 1. Thiết lập thực nghiệm

* **Giao thức OOD Gold Standard**: train trên 67% dữ liệu tải THẤP nhất, test trên 33% tải CAO
  nhất (chưa từng thấy trong huấn luyện) — mô phỏng đúng bài toán zero-shot capacity planning.
* **Hai cách tính sai số** được báo cáo song song để tránh lạc quan giả do làm mượt:
  - *Bucket-average*: gộp tập test thành 8 nhóm theo phân vị workload rồi tính sai số trên
    giá trị trung bình mỗi nhóm (cách làm gốc — làm mượt nhiễu, có thể đánh giá thấp sai số thật).
  - *Full-resolution*: tính trực tiếp trên **toàn bộ** điểm test thô (~21k điểm/service-metric),
    dùng công thức đóng E[Target|do(Workload=w)] = prediction_model.predict(w) (không cần
    Monte Carlo, không mất độ phân giải).
* **Ground-truth direct match** (bằng chứng độc lập mạnh nhất): train/test tách theo **thời
  gian trong từng run** (không gộp nhiều scenario), so khớp trực tiếp với ~408 nghìn điểm đo
  thật trên toàn bộ 90 (scenario × run) của RE2-SS.

---

## 2. Bảng 1 — Độ chính xác theo Service × Metric (bucket vs full-resolution)

| Service | Metric | MAPE bucket (%) | MAPE full-res (%) | R² bucket | R² full-res |
|---|---|---|---|---|---|
| front-end | CPU | 1.65 | 10.16 | -5.275 | -0.022 |
| catalogue | CPU | 43.22 | 51.11 | -1.694 | -0.001 |
| user | CPU | 12.27 | 20.37 | -42.947 | -0.522 |
| carts | CPU | 11.73 | 62.43 | -7.035 | -0.017 |
| orders | CPU | 21.40 | 81.34 | 0.305 | 0.111 |
| payment | CPU | 16.52 | 19.92 | -0.032 | -0.000 |
| shipping | CPU | 11.05 | 81.97 | -0.195 | 0.002 |
| front-end | Memory | 1.81 | 4.70 | -10.994 | -0.011 |
| catalogue | Memory | 3.52 | 3.51 | -422.491 | -0.079 |
| user | Memory | 0.96 | 7.92 | -1.204 | -0.010 |
| carts | Memory | 0.57 | 2.94 | -2.110 | -0.003 |
| orders | Memory | 0.95 | 2.26 | -0.115 | 0.001 |
| payment | Memory | 2.42 | 4.96 | -20.366 | -0.131 |
| shipping | Memory | 0.35 | 1.33 | -1.332 | -0.001 |
| front-end | Socket | 43.42 | 52.96 | -399.137 | -2.908 |
| catalogue | Socket | 1.62 | 13.72 | -0.935 | -0.004 |
| user | Socket | 1.86 | 17.85 | -0.314 | -0.000 |
| carts | Socket | 43.67 | 60.76 | -72.303 | -0.712 |
| orders | Socket | 10.47 | 18.06 | -48.265 | -0.241 |
| payment | Socket | 2.39 | 15.39 | -5.528 | -0.016 |
| shipping | Socket | 5.82 | 8.27 | -77.786 | -0.379 |

**Nhận xét**: MAPE full-resolution thường **cao hơn đáng kể** so với bucket-average — ví dụ rõ
nhất trong Bảng 1: `shipping` CPU đi từ 11.05% (bucket)
lên 81.97% (full-res). Bucket-average từng làm sai số trông tốt
hơn thực tế; nên trích dẫn cột full-resolution làm số liệu chính trong bài báo.

So với bản trước khi ép `LinearRegression(positive=True)` (thay cho `gcm.auto`), MAPE
full-resolution của CPU/Socket ở phần lớn service **tăng** (ví dụ `carts` CPU: 67.53% →
62.43% giảm nhẹ, nhưng `shipping` CPU: 66.64% → 81.97% tăng mạnh) — mô hình ràng buộc đơn
điệu khớp kém hơn cục bộ so với mô hình `gcm.auto` tự do lựa chọn, đổi lại đảm bảo không có
sign-inversion. Memory hầu như không đổi (quan hệ với workload đã gần tuyến tính sẵn).

R² (cả hai cách tính) phổ biến ở mức âm nhẹ đến trung bình ở nhiều service/metric — SCM
không giải thích được phần lớn phương sai ngoài mẫu so với một baseline "đoán trung bình",
dù sai số tuyệt đối (MAPE) vẫn ở mức chấp nhận được cho capacity planning. Cần nêu rõ cả hai
khía cạnh này, không chỉ trích MAPE.

---

## 3. Bảng 2 — Ground-truth direct match (408k điểm, không bucket, toàn bộ 90 run)

| Metric | n điểm | Mean \|err\|% | Median \|err\|% | P90 \|err\|% |
|---|---|---|---|---|
| CPU | 136,496 | 33.45% | 9.05% | 77.76% |
| Latency_p50 | 134,625 | 9.88% | 2.03% | 12.58% |
| Memory | 136,674 | 2.30% | 0.38% | 4.94% |
| Socket | 136,674 | 5.81% | 2.82% | 16.23% |

**Đây là bằng chứng OOD độc lập, không dùng lại tập train/test của Bảng 1.** Với mọi metric,
mean > median (đôi khi gấp 2-4 lần) — nghĩa là phân phối sai số lệch phải rõ rệt: đa số điểm dự
báo tốt nhưng một số run/scenario có sai số rất lớn kéo mean lên cao. Phải báo cáo **cả mean lẫn
median** (và P90 nếu cần thể hiện đuôi), không chỉ chọn một con số đẹp.

---

## 4. Giới hạn cần nêu trong bài báo
1. R² âm ở nhiều service/metric (mục 2) — SCM không phải "gần như hoàn hảo" theo tiêu chuẩn
   giải thích phương sai; giá trị của SCM nằm ở khả năng ngoại suy có kiểm soát + do-calculus,
   không phải R² cao.
2. Sai số CPU cao hơn Memory/Latency một cách hệ thống (xem cả 2 bảng) — nên thảo luận nguyên
   nhân (CPU biến động do nhiều tiến trình đồng thời, khó dự báo tuyến tính hơn workload thô).
3. File nguồn: `test_f1_rmse_evaluation.csv`, `ground_truth_direct_match.csv`,
   `ground_truth_direct_match_summary.csv`.
