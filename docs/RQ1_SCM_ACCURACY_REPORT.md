# RQ1 — Độ chính xác dự báo OOD của SCM (do-calculus)

> **RQ1:** SCM kết hợp do-calculus dự đoán tài nguyên (CPU/Memory/Socket) chính xác đến mức nào
> khi ngoại suy sang vùng tải cao chưa từng quan sát (Out-of-Distribution)?

Tài liệu này sinh **tự động** từ `test_f1_rmse_evaluation.csv` và
`ground_truth_direct_match_summary.csv` — không gõ tay số liệu.

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
| front-end | CPU | 0.75 | 10.19 | -0.294 | -0.001 |
| catalogue | CPU | 27.36 | 11.17 | -0.949 | -0.002 |
| user | CPU | 2.28 | 15.65 | -1.943 | -0.034 |
| carts | CPU | 15.82 | 67.53 | -13.633 | -0.038 |
| orders | CPU | 14.35 | 66.62 | -0.237 | -0.003 |
| payment | CPU | 10.17 | 12.39 | -0.255 | -0.000 |
| shipping | CPU | 9.04 | 66.64 | -0.543 | -0.004 |
| front-end | Memory | 2.27 | 4.99 | -12.269 | -0.034 |
| catalogue | Memory | 0.63 | 3.29 | -5.744 | -0.007 |
| user | Memory | 1.39 | 7.92 | -3.162 | -0.011 |
| carts | Memory | 3.70 | 4.97 | -89.426 | -0.149 |
| orders | Memory | 3.05 | 4.97 | -2.135 | -0.066 |
| payment | Memory | 1.21 | 4.41 | -4.098 | -0.024 |
| shipping | Memory | 0.44 | 1.34 | -3.487 | -0.001 |
| front-end | Socket | 15.35 | 19.03 | -52.462 | -0.174 |
| catalogue | Socket | 9.86 | 11.73 | -65.052 | -0.014 |
| user | Socket | 2.03 | 17.64 | -0.774 | -0.000 |
| carts | Socket | 8.95 | 26.41 | -2.489 | -0.013 |
| orders | Socket | 4.85 | 12.09 | -9.629 | -0.011 |
| payment | Socket | 2.20 | 15.45 | -4.643 | -0.012 |
| shipping | Socket | 0.66 | 6.69 | -0.366 | -0.002 |

**Nhận xét**: MAPE full-resolution thường **cao hơn đáng kể** so với bucket-average — ví dụ rõ
nhất trong Bảng 1: `carts` CPU đi từ 15.82% (bucket)
lên 67.53% (full-res). Bucket-average từng làm sai số trông tốt
hơn thực tế; nên trích dẫn cột full-resolution làm số liệu chính trong bài báo.

R² (cả hai cách tính) phổ biến ở mức âm nhẹ đến trung bình ở nhiều service/metric — SCM
không giải thích được phần lớn phương sai ngoài mẫu so với một baseline "đoán trung bình",
dù sai số tuyệt đối (MAPE) vẫn ở mức chấp nhận được cho capacity planning. Cần nêu rõ cả hai
khía cạnh này, không chỉ trích MAPE.

---

## 3. Bảng 2 — Ground-truth direct match (408k điểm, không bucket, toàn bộ 90 run)

| Metric | n điểm | Mean \|err\|% | Median \|err\|% | P90 \|err\|% |
|---|---|---|---|---|
| CPU | 136,496 | 35.07% | 9.09% | 77.41% |
| Latency_p50 | 134,625 | 9.88% | 2.03% | 12.58% |
| Memory | 136,674 | 2.30% | 0.38% | 5.01% |

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
