# RQ6 — Tổng quát hoá sang hệ thống thứ hai: Train Ticket

> **RQ6:** Kiến trúc MAS-SCM có tổng quát hoá (Plug-and-Play) sang các hệ Microservices khác
> (topology, quy mô khác SockShop) mà không cần hardcode tri thức riêng cho từng hệ hay không?

Sinh tự động từ `trainticket_f1_rmse_evaluation.csv`, `trainticket_model_comparison.csv`,
`trainticket_p_value_statistical_test.csv`, `trainticket_ground_truth_direct_match_summary.csv`,
đối chiếu với kết quả SockShop đã có ở RQ1/RQ2.

---

## 1. Thiết lập
* **Hệ thống thứ hai**: Train Ticket (RCAEval, `phamquiluan/RCAEval` trên Hugging Face) — **28
  microservices**, quy mô lớn hơn SockShop (7 service) gần 4 lần.
* **Dữ liệu**: 30 kịch bản lỗi thật (5 service mục tiêu × 6 loại fault × 1 run), ~21.600 dòng
  viễn trắc giai đoạn bình thường (trước injection), phủ đủ cả 28 service (mỗi kịch bản ghi lại
  toàn bộ hệ thống, không chỉ service bị tiêm lỗi).
* **Cùng chính xác một pipeline đánh giá đã dùng cho SockShop** (full-resolution MAPE, 4 mô hình
  SCM/LinearReg/GradBoost/GaussianProcess, kiểm định Wilcoxon/Friedman tách theo metric, ground-truth
  direct match tách theo thời gian trong từng scenario) — **không dùng lại** script
  `run_trainticket_benchmark.py` cũ trong nhánh gốc (vốn còn lỗi bucket-averaging, thiếu
  GaussianProcess, không có ground-truth match — xem mục 5).
* **Khác biệt duy nhất với SockShop**: chỉ đánh giá CPU và Memory (Train Ticket không có cột
  Socket giống SockShop trong bộ dữ liệu này); Latency không đưa vào 4-model comparison vì baseline
  ML thường không mô hình hoá tốt hành vi bão hoà phi tuyến, giữ nhất quán với cách SockShop tách
  latency ra khỏi phần so sánh MAPE.

---

## 2. Ground-truth direct match: SockShop vs Train Ticket

| Metric | SockShop median \|err\|% | Train Ticket median \|err\|% | SockShop mean | Train Ticket mean |
|---|---|---|---|---|
| CPU | 9.09% | 19.24% | 35.07% | 43.56% |
| Memory | 0.38% | 0.12% | 2.30% | 0.66% |

**Train Ticket có sai số CPU cao hơn SockShop khoảng gấp đôi** (median 19.24% vs 9.09%), nhưng
**Memory lại chính xác hơn** (0.12% vs 0.38%). Không có kết luận đơn giản "tổng quát hoá tốt/xấu"
— mức độ chính xác phụ thuộc metric và có thể phụ thuộc đặc điểm topology/quy mô hệ thống.

## 3. So sánh 4 mô hình trên Train Ticket (MAPE full-resolution)

| Metric | LinearReg | GradBoost | GaussianProcess | SCM_DoWhy |
|---|---|---|---|---|
| CPU | 56.78% | 49.87% | 46.19% | 54.44% |
| Memory | 2.61% | 2.48% | 2.54% | 2.73% |

## 4. Số lần thắng theo (service, metric) — đối chiếu 2 hệ thống

| Model | Số lần thắng / 56 (Train Ticket) | Số lần thắng / 21 (SockShop, RQ2) |
|---|---|---|
| LinearReg | 13 | 1 |
| GradBoost | 20 | 6 |
| GaussianProcess | 19 | 7 |
| SCM_DoWhy | 4 | 7 |

**Phát hiện quan trọng, cần báo cáo trung thực**: Ở SockShop, SCM đồng hạng nhất
(7/21 lần thắng). Ở Train Ticket, **SCM chỉ thắng 4/56 lần —
xếp CUỐI trong 4 mô hình** (GradBoost và GaussianProcess thắng nhiều hơn hẳn). Điều này có nghĩa
**lợi thế độ chính xác điểm của SCM không tổng quát hoá đồng đều giữa các hệ thống** — càng củng cố
luận điểm đã nêu ở RQ2: giá trị cốt lõi của SCM nằm ở khả năng can thiệp do(x), không phải độ
chính xác dự báo thô, vì độ chính xác thô rõ ràng không ổn định qua các topology khác nhau.

## 5. Kiểm định thống kê trên Train Ticket (n=28 service, gấp 4 lần SockShop)

| So sánh | Metric | n | Median Δ | Wilcoxon stat | p-value | Ý nghĩa |
|---|---|---|---|---|---|---|
| SCM_vs_LinearReg | CPU | 28 | -8.21 | 110.0 | 0.15777 | ❌ Không |
| SCM_vs_GradBoost | CPU | 28 | 0.105 | 124.0 | 0.07354 | ❌ Không |
| SCM_vs_GaussianProcess | CPU | 28 | 1.575 | 118.0 | 0.05338 | ❌ Không |
| SCM_vs_LinearReg | Memory | 28 | -0.005 | 171.5 | 0.91906 | ❌ Không |
| SCM_vs_GradBoost | Memory | 28 | 0.0 | 18.5 | 0.03242 | ✅ Có |
| SCM_vs_GaussianProcess | Memory | 28 | -0.01 | 118.0 | 0.5429 | ❌ Không |
| SCM_vs_LinearReg | ALL_COMBINED | 56 | -0.035 | 461.5 | 0.05892 | ❌ Không |
| SCM_vs_GradBoost | ALL_COMBINED | 56 | 0.005 | 255.5 | 0.01425 | ✅ Có |
| SCM_vs_GaussianProcess | ALL_COMBINED | 56 | 0.0 | 459.0 | 0.05585 | ❌ Không |

Đa số so sánh không có ý nghĩa thống kê. `SCM_vs_GradBoost` có ý nghĩa ở Memory và ALL_COMBINED
nhưng **median Δ gần như bằng 0** (0.000–0.005 điểm %) — có ý nghĩa thống kê nhưng **không có ý
nghĩa thực tiễn** (practical significance), một ví dụ điển hình cần phân biệt rõ trong bài báo.

---

## 6. Giới hạn phương pháp luận
1. **Chỉ 1 run/scenario** (30 kịch bản, không phải 90 như SockShop) — do người dùng chọn phạm vi
   nhỏ để có kết quả nhanh. Có thể mở rộng lên 90 kịch bản (`download_trainticket_data.py --all`)
   để tăng độ tin cậy nếu cần cho bản nộp cuối.
2. Script gốc `run_trainticket_benchmark.py` trong nhánh `TrainTicket` (đã merge vào repo nhưng
   KHÔNG được dùng để sinh số liệu ở đây) có cùng lỗi bucket-averaging + thiếu GaussianProcess đã
   từng mắc phải ở SockShop trước đợt rà soát này — **không trích dẫn file
   `03_trainticket_scm_evaluation.csv` cũ**, chỉ dùng các file `trainticket_*` mới sinh ra từ
   `src/scm/trainticket_evaluation.py`.
3. Chưa kiểm tra ngoại suy cực đoan (do(workload) rất lớn) trên Train Ticket như đã làm ở SockShop
   (mục extrapolation_suspect) — nằm ngoài phạm vi RQ6 lần này, chỉ tập trung câu hỏi tổng quát
   hoá độ chính xác OOD cơ bản.

File nguồn: `trainticket_f1_rmse_evaluation.csv`, `trainticket_model_comparison.csv`,
`trainticket_p_value_statistical_test.csv`, `trainticket_ground_truth_direct_match_summary.csv`.
