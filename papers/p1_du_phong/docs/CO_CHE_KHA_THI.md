# Cơ chế của vi phạm SLO khi thêm tính năng — kết quả đã chốt

> Tái lập: `python papers/p1_du_phong/experiments/model_eval/mechanism_analysis.py [--part signal|control|burst|auc]`
> Dữ liệu: 1 318 bậc tải / 136 run / 19 cấu hình tính năng (Sock Shop, trần `RE2`), huấn luyện
> cơ chế **chỉ** trên `measured_data.load('baseline')` (2 704 dòng đo hệ bình thường, 0 dòng có tính năng).

## Câu hỏi

Đại lượng nào **quyết định** tính khả thi khi thêm một tính năng hướng người dùng, và đại lượng
nào **dự phóng được** trước khi tính năng tồn tại? Hai tập này có trùng nhau không?

---

## 1. Chỉ CPU là tài nguyên dự phóng được

`R²` của `<metric> ~ workload` trên dữ liệu đo hệ bình thường (§S1):

| | cpu | mem | socket | diskio | error | latency-50 | latency-99 |
|---|---|---|---|---|---|---|---|
| front-end | **0,972** | 0,042 | 0,243 | — | 0,000 | 0,158 | 0,097 |
| catalogue | **0,987** | 0,001 | 0,000 | 0,000 | — | 0,048 | 0,007 |
| user | **0,988** | 0,001 | 0,004 | — | — | 0,024 | 0,005 |
| carts | 0,780 | 0,018 | 0,017 | — | — | 0,001 | 0,027 |
| orders | 0,567 | 0,031 | 0,005 | 0,000 | — | 0,003 | 0,052 |

Ba điều kiện để một tài nguyên dự phóng được: **có tín hiệu theo tải**, **có trần đọc được từ
telemetry**, **là thứ làm vỡ SLO**. Telemetry chuẩn (cAdvisor) chỉ có
`container-spec-cpu-quota` — không có cột trần cho mem/socket/disk. Chỉ CPU thoả cả ba.

## 2. Ngưỡng toàn cục trên mức sử dụng CPU **không dùng được**

- **88%** bậc vi phạm SLO có `u_max < u* = 0,8783`
- Dùng `u` **đo được thật** (sai số dự đoán = 0), ngưỡng tốt nhất đạt **76,9%**, hơn baseline
  hằng số "luôn FEASIBLE" chỉ **1,2 điểm**
- Ở mức **không có FEASIBLE sai nào**, ngưỡng trên `u` chỉ trả lời được **0,8%** số bậc

SLO vỡ khi CPU còn xa trần. Đây là kết quả vững nhất của phân tích.

## 3. Trần CPU là nguyên nhân — bằng chứng can thiệp

Đối chứng âm: `SS-TRAIN` thu **không có trần** (không thể throttle) vs ramp có trần `RE2`, ở
**cùng tải gateway** (§S2):

| tải (req/s) | p99 không trần | p99 trần RE2 | tỉ lệ | socket tỉ lệ | CPU tỉ lệ |
|---|---|---|---|---|---|
| 50 | 0,0234 s | 0,0247 | 1,05× | 1,00× | 1,05× |
| 150 | 0,0243 | 0,0889 | **3,66×** | 1,42× | 1,00× |
| 200 | 0,0247 | 0,1880 | **7,61×** | 2,26× | 0,97× |
| 250 | 0,0254 | 0,2314 | **9,12×** | 1,67× | 0,94× |
| 275 | 0,0319 | 0,1579 | 4,95× | **3,31×** | 0,89× |

**Bỏ trần thì hiện tượng vỡ p99 biến mất**: p99 phẳng 23–32 ms suốt dải 0–275 req/s.
CPU mỗi request **như nhau** (tỉ lệ 0,97–1,06 tới 175 req/s) → không phải ứng dụng nặng hơn.

`p50` gần như không đổi (≤3,1× trong khi p99 đổi 9,1×) → **p50 là phân vị sai** để đo hiện tượng này.

> **Nhiễu còn lại:** `SS-TRAIN` là bậc tải tĩnh, tập có trần là ramp; hai phiên thu khác ngày;
> `SS-TRAIN` không có nhãn vi phạm nên chỉ so được p99/socket. Cần **X1** (dưới đây) để loại.

## 4. Throttling chứa thông tin mà mức sử dụng **không** chứa

Sức phân biệt bậc vi phạm, **6 đặc tả** (đơn vị × nhãn × phạm vi) (§S4):

| đặc tả | `u` đo | `u` **dự đoán** | **throttle đo** | throttle suy từ `u` |
|---|---|---|---|---|
| quyết định · bất kỳ · tính năng | 0,835 | 0,813 | **0,888** | 0,842 |
| quyết định · **đa số** · tính năng | 0,689 | 0,676 | **0,784** | 0,700 |
| (run, bậc) · tính năng | 0,745 | 0,721 | **0,833** | 0,748 |
| quyết định · bất kỳ · gồm base | 0,847 | 0,833 | **0,896** | 0,855 |
| quyết định · đa số · gồm base | 0,687 | 0,676 | **0,785** | 0,700 |
| (run, bậc) · gồm base | 0,767 | 0,746 | **0,852** | 0,770 |

Bootstrap trên **hiệu số** AUC (cùng mẫu), **18/18 phép kiểm** có CI loại trừ 0:

| so sánh | hiệu AUC (khoảng qua 6 đặc tả) | `P(d≤0)` |
|---|---|---|
| throttle đo − `u` đo | **+0,049 … +0,097** | ≤ 0,0000 |
| **throttle đo − throttle suy từ `u`** | **+0,041 … +0,085** | ≤ 0,0005 |
| throttle đo − `u` dự đoán | +0,063 … +0,113 | ≤ 0,0003 |

Dòng giữa là dòng quyết định: phần hơn của throttling **không lấy được bằng bất kỳ phép biến đổi
đơn điệu nào của `u`** (isotonic per-container, fit ngoài fold). Nên throttling không phải
"`u` viết cách khác".

---

## Giả thuyết đã bác bỏ — giữ lại để không ai làm lại

### B1. Độ bùng nổ là cơ chế — **bác bỏ**

| | AUC → vi phạm |
|---|---|
| `cv` số request mỗi 100 ms | **0,504** |
| `burst_p99` | **0,516** |
| (đoán ngẫu nhiên) | 0,500 |

`ρ(throttle, bùng nổ)` đảo dấu giữa các dải tốc độ (+0,215 ở Q3 → **−0,640** ở Q4) → nhiễu,
không phải cơ chế. Throttle tương quan với **tốc độ trung bình** (+0,444).

**Phạm vi bác bỏ:** biến đo được là **phân tán lượt đến** (số dòng log mỗi 100 ms). CFS throttle
theo **nhu cầu CPU** trong chu kỳ (mức song song × thời gian phục vụ) — một container 4 thread ×
12 ms CPU có thể đốt hết quota 50 ms từ **một** lượt. Bản *"bùng nổ theo mức song song"* **chưa được kiểm**.

### B2. "Có cơ chế thứ ba ngoài CPU và throttling" — **bác bỏ**

Khi chỉ tính throttle trên 5 app service, 4% vi phạm không giải thích được. Đưa **DB và hạ tầng**
vào (`carts-db`, `orders-db`, `user-db`, `queue-master`) thì khoảng trống đóng: `thr::carts-db`
= 0,125 ở nhóm đó vs **0,000** ở nhóm đạt SLO. Không cần cơ chế thứ ba — cần **đủ container**.

---

## Ba sai sót phương pháp đã sửa (ai làm lại phải giữ)

1. **Đơn vị phân tích.** Cùng chỉ số `u_max` cho AUC 0,745 hay 0,835 tuỳ cách gộp dòng. Đơn vị
   đúng = **một quyết định** = `(tính năng, mức tải)`; nhiều run của cùng cấu hình là đo lặp lại.
2. **Rò rỉ.** Isotonic `u → throttle` fit rồi đánh giá trên cùng dữ liệu cho AUC 0,912 lạc quan.
   Phải `GroupKFold` theo tính năng.
3. **Phép kiểm sai.** So hai AUC bằng cách xem hai khoảng tin cậy có chồng nhau không là sai khi
   hai chỉ số tính trên **cùng mẫu**. Phải bootstrap trên **hiệu số**.

## Giới hạn

1. **Một hệ thống.** `trainticket` và `RE2-OB` chỉ có kịch bản tiêm lỗi, **không có run nào mang
   nhãn tính năng**. Tuyên bố về phán quyết chỉ có bằng chứng trên Sock Shop / 16 tính năng.
2. **Throttle và socket đo đồng thời** với vi phạm → chúng là **bộ phát hiện**, chưa chứng minh là
   **biến dự phóng**. `u` là đại lượng duy nhất tính được trước khi cài đặt.
3. `catalogue-db`, `rabbitmq`, `edge-router` **không có trần** trong cấu hình `RE2` → không thể
   throttle. Kết quả có thể phụ thuộc chính lựa chọn đặt trần đó.
4. Node nghẽn là `front-end` ở **97,7%** quan sát → trên testbed này, đóng góp của đồ thị phụ
   thuộc là rất mỏng, và baseline hằng số "đoán gateway" đạt 97,7%.

## Thí nghiệm còn cần

### X1 — đổi **chu kỳ** CFS, giữ nguyên hạn ngạch *(quyết định)*

```bash
docker update --cpu-period=10000  --cpu-quota=5000   front-end   # 0,5 core, chu kỳ 10 ms
docker update --cpu-period=100000 --cpu-quota=50000  front-end   # 0,5 core, mặc định
docker update --cpu-period=500000 --cpu-quota=250000 front-end   # 0,5 core, chu kỳ 500 ms
```

Cả ba **cùng 0,5 core**. Nếu throttling là cơ chế, điểm gãy phải dịch chuyển rõ. Nếu không dịch →
mệnh đề bị bác. Đây là cách duy nhất loại hết nhiễu của §3 (cùng phiên, cùng ramp, cùng dung lượng).

### X2 — cùng `λ` trung bình, khác **mức song song**

Bản bùng nổ còn bỏ ngỏ (xem B1): cùng tốc độ request nhưng khác số kết nối đồng thời.

---

## Cách chạy X1 (cần máy có Docker + Sock Shop đang chạy)

Đã chuẩn bị sẵn:

- `deploy/sockshop/limits.json` — thêm `RE2-P10`, `RE2-P100`, `RE2-P500`: **cùng số core như `RE2`**,
  chỉ khác `_period_us` (10 000 / 100 000 / 500 000 µs).
- `papers/p1_du_phong/experiments/collect/load_sweep_collect.py:apply_limits` — nhận `_period_us` và đặt trần bằng
  `--cpu-period` + `--cpu-quota`, đọc lại `CpuPeriod`/`CpuQuota` để xác minh. Cấu hình **không có**
  `_period_us` đi đúng đường cũ (`--cpus`), không đổi một dòng nào.

```bash
for P in RE2-P10 RE2-P100 RE2-P500; do
  python papers/p1_du_phong/experiments/collect/load_sweep_collect.py --ramp --limits $P \
      --features base --repeats 3 --out-dir data/raw/SS-X1-$P
done
```

Ba cấu hình **cùng 0,5 core** cho `front-end`. Chỉ khác dung sai bùng nổ.

### Đọc kết quả

| | nếu "khả thi = ngưỡng trên mức sử dụng" | nếu **throttling** là cơ chế |
|---|---|---|
| `u` tại **điểm gãy** | **giống nhau** ở cả ba chu kỳ | **khác nhau** rõ |
| điểm gãy `R*` | giống nhau | dịch chuyển theo chu kỳ |

Đây là điểm mấu chốt: cả ba có **cùng dung lượng CPU trung bình**, nên mọi mô hình dựa trên mức sử
dụng bắt buộc dự đoán cùng một điểm gãy. Nếu điểm gãy dịch chuyển thì mô hình mức sử dụng bị bác
bằng can thiệp, không còn là suy luận tương quan.

Lưu ý khi đọc: chu kỳ dài **giảm tỉ lệ** throttle nhưng **tăng thời gian mỗi lần đóng băng**
(tới 250 ms ở chu kỳ 500 ms). Nên kỳ vọng là p99 **xấu nhất ở chu kỳ 500 ms** dù throttle ít hơn —
hai hiệu ứng ngược chiều, và `u` tại điểm gãy là đại lượng đọc sạch nhất.

### Hai rủi ro chưa kiểm được (máy phân tích không có Docker)

1. **Xung đột `NanoCpus`.** Docker có thể từ chối `--cpu-period` khi container đã bị `--cpus` đặt
   `NanoCpus`. Hàm `_apply_period_quota` phát hiện lỗi đó, in hướng dẫn tạo lại container bằng
   compose (`cpu_period`/`cpu_quota`) rồi **thoát** — không bao giờ chạy tiếp với trần sai.
2. **Manifest thêm một khoá.** `CURRENT_LIMITS` giờ có `period_us` (giá trị `null` với cấu hình cũ),
   nên manifest của lần chạy mới khác lần cũ **một khoá null**. Không ảnh hưởng con số nào.
