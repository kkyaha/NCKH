# RQ4 — Giá trị của lan truyền qua đồ thị phụ thuộc thật (Tầng 1)

> **RQ4:** Lan truyền workload qua đồ thị phụ thuộc THẬT giữa các service (Tầng 1: Workload→Workload)
> có dự báo chính xác hơn giả định đơn giản "mọi service hạ nguồn đổi cùng % với front-end"
> (naive/delta-đều) hay không?

Sinh tự động từ `rq4_propagation_value_test.csv`.

---

## 1. Thiết lập
* Cùng giao thức OOD (train 67% tải thấp → test 33% tải cao) trên `front-end_workload`.
* **SCM Tier-1**: hồi quy tuyến tính hệ số không âm (`LinearRegression(positive=True)`) từ
  (các) node cha THẬT theo topology `sockshop_agent_graph.json` (vd `orders_workload` phụ
  thuộc `front-end_workload`; `carts_workload`/`user_workload` phụ thuộc CẢ `front-end_workload`
  LẪN `orders_workload`).
* **Naive baseline**: giả định workload hạ nguồn thay đổi theo đúng % thay đổi của
  `front-end_workload` so với baseline train, không dùng thông tin đồ thị.
* Cả hai được so với workload THẬT đo cùng thời điểm trong tập test (không phải giả lập).

## 2. Kết quả

| Service | Parent(s) thật | SCM MAPE (%) | Naive MAPE (%) | Thắng |
|---|---|---|---|---|
| catalogue | ['front-end_workload'] | 5.25 | 5.35 | SCM |
| user | ['front-end_workload', 'orders_workload'] | 7.02 | 13.24 | SCM |
| carts | ['front-end_workload', 'orders_workload'] | 7.75 | 7.56 | Naive |
| orders | ['front-end_workload'] | 19.35 | 21.08 | SCM |
| payment | ['orders_workload'] | 9.42 | 19.76 | SCM |
| shipping | ['orders_workload'] | 8.35 | 20.77 | SCM |

**SCM thắng 5/6 node** theo MAPE. Kiểm định Wilcoxon (n=6): stat=2.00, p=0.09375 (❌ chưa đạt ngưỡng 0.05).

## 3. Diễn giải trung thực
* Với n chỉ = 6 (số service không phải gốc trong topology 7-service), kiểm định thống
  kê có lực kiểm định (power) thấp — không nên tuyên bố "có ý nghĩa thống kê" trừ khi p<0.05
  thực sự đạt được ở lần chạy cụ thể.
* Hiệu ứng thực tế (effect size) lớn ở nhiều node (`payment`, `shipping`, `user`: SCM giảm MAPE
  gần một nửa so với naive) — đây là bằng chứng **định hướng mạnh** dù n nhỏ.
* `carts` là ngoại lệ (naive thắng nhẹ) — đáng điều tra thêm (có thể do `carts_workload` phụ
  thuộc phi tuyến hoặc có yếu tố nhiễu khác ngoài 2 parent đã biết).
* Đây là RQ **mới được xây dựng và chạy lần đầu** trong đợt rà soát này — trước đó RQ4 ở trạng
  thái "chưa có baseline" (xem `UNIFIED_REFERENCE_DOC.md`). Nên chạy lại với nhiều seed/giao
  thức chia dữ liệu khác nhau trước khi đưa vào bản thảo cuối để tăng độ tin cậy.

File nguồn: `rq4_propagation_value_test.csv`.
