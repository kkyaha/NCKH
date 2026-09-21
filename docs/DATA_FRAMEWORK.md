# KHUNG CƠ CHẾ VÀ DỮ LIỆU — chốt trước khi thu thập

> Phạm vi: **chỉ** định nghĩa bài toán, hợp đồng dữ liệu và kế hoạch thu. **Không** tối ưu mô hình
> (làm sau khi có dữ liệu). Các mục đánh dấu ⚠ là giả định CHƯA kiểm chứng.
> Đọc kèm: `docs/HANDOFF.md` (bối cảnh), `experiments/load_sweep_collect.py --help`.

---

## 1. Bài toán (một câu)

Cho một **yêu cầu tính năng mới chưa từng có trong hệ thống**, dự đoán hệ thống Sock Shop
**có còn đáp ứng SLO không** ở tải đỉnh kỳ vọng, và node nào sẽ nghẽn trước.

Không phải dự báo chuỗi thời gian; là **suy luận can thiệp (what-if)** trên hệ chưa có tính năng đó.

### Đầu vào (hợp đồng)

| Trường | Ý nghĩa | Nguồn |
|---|---|---|
| `requirement` | mô tả ngôn ngữ tự nhiên | người dùng |
| `L_peak` | tải NỀN đỉnh kỳ vọng tại gateway (req/s) | **tường minh**, không lấy trung bình của dữ liệu train |
| `lambda_star` | request của tính năng / giây, tính = `anchor% × L_peak` | mặc định từ archetype (`expected_delta_pct`), người dùng ghi đè |
| `chain` | tập service bị gọi (+ bội số `k_s`, mặc định 1) | `CALL_CHAINS` (đã có) |
| `C_s` | trần CPU mỗi node chính, cùng đơn vị `_cpu` (= 100 × số core) | cấu hình triển khai (K8s limit) |
| `SLO` | p99 độ trễ, tỉ lệ lỗi | `deploy/sockshop/slo.json` |

### Đầu ra

Mỗi node chính: CPU dự đoán (khoảng) và mức sử dụng `u_s = CPU'_s / C_s`.
Chung: `FEASIBLE | MARGINAL | INFEASIBLE`, node nghẽn dự đoán, và **cờ ngoại suy** (tách riêng khỏi phán quyết).

### Nhãn thực tế

Một điểm (tính năng, cường độ, `L`) là **khả thi** nếu SLO được giữ ở tải đó, đo phía người dùng:
`p99 ≤ slo.p99_s` và `tỉ lệ lỗi (5xx + timeout) ≤ slo.err_rate`, tính trên 60% cuối của mỗi bậc tải.

**Điểm gãy** = bậc đầu của **chuỗi vi phạm liên tiếp cuối cùng** (ramp dừng khi vi phạm liên tiếp). Một bậc vi phạm rồi bậc sau lại đạt SLO
(GC, nhiễu) là *thoáng qua*, được ghi riêng, không phải điểm gãy. Định nghĩa "bậc vi phạm đầu tiên" ban đầu quá dễ vỡ vì một bậc nhiễu:
Phase 0 ramp thứ ba vi phạm ở 140 req/s trong khi hai ramp kia đạt SLO tới 220. Độ phân giải điểm gãy = bước tải (20 req/s).

## 2. Cơ chế dự đoán (mô tả; chưa tối ưu)

1. **Học cơ chế từ dữ liệu KHÔNG có tính năng**: `CPU_s = g_s(W_s)` cho từng service (NNLS/elasticity), và lan truyền
   workload gateway → con theo cơ cấu request nền.
2. **Can thiệp = thêm một lớp request** (không phải `gateway +Δ%`):
   `W_s' = W_s(nền tại L_peak) + λ*·k_s` với `s ∈ chain`; phần request tính năng **không** lan ra ngoài chain.
   CPU của gateway tính theo TỔNG request (nền + tính năng).
3. **Phán quyết**: `u_s = CPU'_s / C_s`; ngưỡng MARGINAL/INFEASIBLE **hiệu chỉnh từ điểm gãy của baseline**
   (Phase 0), không đặt tay. ⚠ Mô hình đơn giản giả định SLO vỡ quanh `u ≈ 0,85`; phải đo.
4. **Độ tin cậy** (tách khỏi phán quyết): nếu dự đoán vượt vùng dữ liệu train (P99, guard G7) → cờ ngoại suy/`UNDECIDED`.

Mã hiện tại lệch khung ở 4 điểm (KHÔNG sửa ở giai đoạn này, sửa sau khi có dữ liệu):
`capacity_agent.py:724` (baseline = trung bình gộp), `:1351` (trần = P99 của train),
`parser_agent.py:451` (Δ = prior archetype), `assess_capacity:1491` (chain chỉ là mặt nạ chọn phạm vi).

## 3. Phạm vi node

**Chấm điểm**: `front-end, catalogue, user, carts, orders`. `payment`, `shipping` vẫn nằm trong mô hình để lan
truyền tải nhưng không chấm (CPU quá nhỏ làm sai số tương đối vô nghĩa; shipping bất đồng bộ, tín hiệu yếu).
Quy tắc chọn (áp trên dữ liệu baseline TRƯỚC khi xem đáp án): qua cổng tín hiệu (R² ≥ 0,3 + negative control),
CPU nền ≥ 0,5 điểm % ở 100 req/s, và nằm trên đường đồng bộ của yêu cầu người dùng.
Ngoài phạm vi chấm điểm: `mem`, `latency` của từng node, node DB/hàng đợi (`socket`, `diskio` vẫn được thu cho đủ bộ chuẩn nhưng không chấm).

## 4. Hợp đồng đơn vị và cột (`simple_metrics.csv`)

| Cột | Đơn vị | Ghi chú |
|---|---|---|
| `<svc>_workload` | req/s | từ `/metrics` thật, đã loại route `metrics` |
| `<svc>_cpu` | **% của 1 core** (100 = 1 core) | delta bộ đếm cgroup / delta thời gian |
| `<svc>_mem` | byte | working set |
| `<svc>_socket` | số | socket đang mở của container (`/proc/<pid>/net/sockstat`; cần `--pid=host`) |
| `<svc>_diskio` | byte/s | đọc + ghi (blkio) |
| `<svc>_latency-50/90/95/99` | giây | từ histogram, có độ phân giải theo bucket |
| `<svc>_error` | req/s | status ≥ 500 |
| `vm_cpu_util` | 0–1 | phát hiện host bão hoà |
| `time` | epoch giây | cửa sổ lấy mẫu `--interval` (khuyến nghị 3–5 s) |
| nhãn | `load_level`, `feature`, `feature_pct`, `limits_cfg` (+ ramp: `step_idx`, `target_rps`, `step_warm`, `step_violated`) | |

Đơn vị CPU: đối chiếu CPU/(req/s) giữa RE2-SS cũ và dữ liệu mới cho cùng cỡ (0,6–1,1×) nên `_cpu` của RE2-SS
**rất có khả năng cũng là % của một core**; nhận định "core/byte" trong docstring `compute_headroom` và README nhiều khả năng sai (chưa sửa).

## 4b. Chuẩn telemetry (tương thích RE2-SS/RCAEval)

Nguyên tắc: bộ dữ liệu phải **trông như telemetry của một hệ thống thật**; phần duy nhất thêm vào là **ground truth
khi có tính năng mới**. Mỗi run có bốn tệp chuẩn và một lớp nhãn tách riêng.

| Tệp mỗi run | Nội dung | Tương đương chuẩn |
|---|---|---|
| `metrics.csv` | bộ đếm tích lũy, cột `<svc>_container-…` giống RE2-SS, gồm `spec-cpu-quota` (**giới hạn CPU**) | cAdvisor/Prometheus: `container_cpu_usage_seconds_total`, `container_spec_cpu_quota`, `container_sockets`… |
| `simple_metrics.csv` | dẫn xuất theo cửa sổ: `cpu, mem, socket, diskio, workload, error, latency-50/90/95/99` | RE2-SS `simple_metrics.csv` |
| `logs.csv` | **mọi** dòng log, cột `time,timestamp(ns),container_name,message,level,req_path,error` | RE2-SS `logs.csv` |
| `routes.csv` | tốc độ theo `(method, route, status_class)` | RED theo từng endpoint (phần ta thêm) |
| cột `gt_*`, `steps.json`, manifest | ground truth (tính năng, cường độ, trần, bậc tải, vi phạm SLO, điểm gãy) | phần ta thêm |

Ánh xạ sang Prometheus: `_cpu = 100·rate(container_cpu_usage_seconds_total[w])`, `_mem = container_memory_working_set_bytes`,
`_socket = container_sockets`, `_diskio = rate(fs_reads_bytes + fs_writes_bytes)`, `_workload = rate(request_duration_seconds_count[w])`,
`_latency-q = histogram_quantile(q, rate(…_bucket[w]))`, `_error = rate({status=~"5.."})`.

**Khác RE2-SS (ghi vào Threats to Validity):** số liệu ứng dụng lấy từ `/metrics` của từng service (RE2-SS dùng Istio);
không có traces (Sock Shop tắt Zipkin); một máy Docker thay vì K8s nhiều node; timestamp log do Docker daemon gắn.

**Lớp ground truth** là toàn bộ cột `gt_*` cộng `steps.json`. Bỏ chúng đi thì còn đúng telemetry sản xuất, tức đầu vào
mà bộ dự đoán sẽ gặp khi triển khai. Cưỡng chế bằng `experiments/data_contract_check.py` (mục "tuân thủ schema chuẩn").
Đã kiểm chứng: **RE2-SS thật đạt 0 FAIL** (chỉ thiếu `routes.csv`), run dựng bằng harness đạt 0 FAIL / 0 WARN (offline).

Phát hiện khi đối chiếu: `metrics.csv` của RE2-SS **có giới hạn CPU** (`container-spec-cpu-quota`), nên nhận định trong
`HANDOFF.md` ("không dataset nào có K8s limit") **không đúng với RE2-SS**. Trần `C_s` là telemetry chuẩn đọc được
từ cAdvisor, không phải đầu vào tự chế.

## 5. Ma trận thu thập

Thư mục gốc **tách riêng** để loader không trộn dữ liệu có tính năng vào train.

| Pha | Mục đích | Lệnh (rút gọn) | Gốc dữ liệu | Ước tính |
|---|---|---|---|---|
| **0** hiệu chuẩn | xác nhận trần RE2, chốt SLO, kiểm tính bất biến của CPU/request khi có trần | `--ramp --limits RE2 --features base --repeats 3` | `SS-LIMITS` | ~40 ph |
| **A** huấn luyện | học cơ chế; KHÔNG tính năng, KHÔNG trần | `--levels 10,25,50,75,100,150,200,250 --repeats 2 --hold 240 --warmup 45` | `SS-TRAIN` | ~75 ph |
| — đóng băng | ghi dự đoán P0/P1/P1_ctrl (hash) từ Pha A + Phase 0 **trước** khi chạy B | `experiments/freeze_predictions.py` | `data/processed/frozen/predictions_frozen_<cfg>.json` | vài giây |
| **B** đáp án (chỉ cấu hình `RE2`, hợp lệ duy nhất) | điểm gãy VÀ mức sử dụng từng node theo (tính năng × cường độ) | `--ramp --limits RE2 --features base,promo,recs,track,review --feature-scales 1,2 --repeats 2 --interval 3 --cooldown 60` | `SS-LIMITS` | ~2,5 h |

Tổng ≈ 4,5 giờ (Phase 0 + Pha A đã xong; còn lại Pha B ~2,5 giờ). Các pha độc lập; mỗi lượt ghi ngay khi xong nên dừng giữa chừng không mất dữ liệu.

Cấu hình trần **chính = `RE2`** (`deploy/sockshop/limits.json`): giới hạn K8s thật của RE2-SS đọc từ `container-spec-cpu-quota`
(run `carts_cpu/1`): 0,5 core cho carts, catalogue, front-end, orders, shipping, queue-master và các DB; `user` 0,2; `payment` 0,1.
⚠ RE2-SS **thay đổi giới hạn theo kịch bản** (một số run có 0,8 / 0,4), ta lấy giá trị cơ sở. Với 0,5 core và ~0,2 điểm % mỗi req/s,
front-end chạm trần ở khoảng 250 req/s, trong tầm loadgen, nên cấu hình này vừa chuẩn vừa khả thi.
Cấu hình phụ `C1` (⚠ sơ bộ; tính từ hệ số đo thử 5 mẫu/ô) siết trần để điểm gãy rơi vào ~77–191 req/s và nghẽn ở ≥ 3 node khác nhau;
chỉ dùng nếu `RE2` cho điểm gãy quá giống nhau giữa các tính năng. `review` được kỳ vọng KHÔNG đổi điểm gãy so với baseline (ca đối chứng chống báo động giả).

## 5b. Kết quả Phase 0 (2026-09-21, trần `RE2`, 3 ramp baseline, dữ liệu `SS-LIMITS`)

| Đo | Kết quả |
|---|---|
| Điểm gãy baseline | đạt SLO đến **220**, vi phạm bền vững từ **240 req/s**, giống nhau ở cả 3 ramp |
| Nhiễu thoáng qua | 1 bậc vi phạm ở 140 req/s (277 ms) rồi đạt lại; xử lý đúng bởi định nghĩa điểm gãy bền vững |
| Độ nhạy theo SLO p99 | 150–500 ms cho cùng điểm gãy (220/240); 100 ms thì lệch (dịch xuống 140–200); 1 s thì 240–260 ⇒ **chốt 250 ms, lỗi 1%** |
| Node nghẽn | **front-end** (87% trần 0,5 core ở 220 req/s); các node khác ≤ 34% (carts 34, user 28, catalogue 16, orders 11) |
| SLO vỡ ở mức sử dụng | ≈ 87% ⇒ giả định `u ≈ 0,85` trong mục 2 **được xác nhận sơ bộ** (từ 3 ramp, chỉ front-end) |
| CPU/request khi có trần so với không trần | front-end 1,01×, catalogue 1,03×, user 1,05×, carts 1,05×, payment 1,07× (bất biến trong ≤ 7%); orders 0,89× và shipping 0,49× chưa so được sạch vì dữ liệu không trần dùng để so là từ loadgen cũ (tài khoản dùng chung, ~9% đơn bị từ chối) ⇒ **so lại với Pha A** |

Hệ quả: ở trần RE2 baseline chỉ nghẽn ở front-end, nên bài toán "node nào nghẽn" chỉ có thông tin khi tính năng nặng
(recs) chuyển nghẽn sang orders/catalogue; nếu Pha B cho điểm gãy quá giống nhau giữa các tính năng thì dùng cấu hình phụ `C1`.
Độ phân giải điểm gãy = 20 req/s (~9%). Bậc quá điểm gãy (260) là hệ sập (p99 30 giây), không dùng để đọc số.

## 5c. Sức phân biệt của thiết kế (tính TRƯỚC khi chạy Pha B, từ mô hình dev fit trên dữ liệu Phase 0)

Dưới trần `RE2` front-end là nút nghẽn của gần như mọi ô, mà P0, P1 và P1_ctrl tính tải front-end giống hệt nhau (chain luôn chứa gateway), nên:
P0 khác P1 ở **1/8** ô tính năng, P1 khác chain sai ở **1/8** ô ⇒ Pha B chỉ dưới RE2 **không chứng minh được giá trị của chain**.
Vì vậy cần một cấu hình PHÂN BIỆT bên cạnh `RE2` (cấu hình CHUẨN để so sánh với RE2-SS).

**C1 và C2 đều thất bại vì CFS throttling ở hạn ngạch nhỏ (Phase 0b/0c, 2026-09-21):**

| Cấu hình | Node bị đẩy | Kết quả |
|---|---|---|
| C1 | `carts` (JVM) 0,15 core | dùng **95% trần ở 40 req/s** (mô hình tuyến tính: 19%), p99 10 giây, 14,7% lỗi; vi phạm SLO từ bậc đầu ở **3/3** ramp |
| C2 | `catalogue` (Go) 0,08 core | SLO vỡ ở **140 req/s** thay vì ~223 dự đoán, khi CPU trung bình mới **41–49% trần**; tỉ lệ chu kỳ CFS bị throttle **0% (≤80) → 5% (120) → 18% (140) → 32% (160)**; p99 phía server của catalogue 5 → 99 → 464 ms (99 ms ≈ một chu kỳ CFS) |

Kết luận: với hạn ngạch nhỏ, độ trễ do **throttling do đợt tải song song** quyết định, không phải mức sử dụng trung bình. Tiền đề "SLO vỡ ở u\* ≈ 0,88 bất kể node" chỉ được
kiểm chứng ở hạn ngạch ≥ 0,5 core (front-end dưới RE2). Muốn nghẽn ở node khác phải dùng hạn ngạch nhỏ, tức vào chế độ throttling, **ngoài phạm vi bộ dự đoán hiện tại**.
Bài học cho Threats to Validity: giới hạn CPU nhỏ (và node JVM nói riêng) phải được kiểm chứng bằng đo, không suy từ hệ số tuyến tính ở tải nhẹ.
Collector nay ghi `container-cpu-cfs-throttled-*` (metric chuẩn cAdvisor) nên chẩn đoán được trực tiếp.

**Hệ quả cho cách đánh giá giá trị của chain.** Không thể dựa vào việc "các bộ dự đoán cho node nghẽn khác nhau" (dưới RE2, front-end luôn nghẽn trước vì mỗi req/s tốn ~0,2 điểm % ở front-end
so với ~0,03 ở catalogue; không có hạn ngạch hợp lệ nào đẩy node khác tới trần trong tầm tải ≤ 250 req/s). Thay vào đó dùng phép đo **không cần bão hoà**:
so **mức sử dụng dự đoán từng node** `u_s` của P0/P1/P1_ctrl với mức đo được ở từng bậc tải trước điểm gãy của ramp tính năng. Ở đây P0 và P1 khác nhau rất xa
(ví dụ orders: P0 +0,7 req/s so với P1 +18 req/s cho promo), nên chain có thể được chứng minh hoặc bác bỏ ngay cả khi front-end nghẽn ở mọi ô.
`predictions_frozen_*.json` lưu `u` từng node theo lưới tải để làm việc này. Quyết định cấp hệ thống (điểm gãy, node nghẽn) vẫn được báo cáo nhưng chủ yếu phân biệt qua front-end.

## 6. Chia dữ liệu và chống rò rỉ

| Vai trò | Dữ liệu | Quy tắc |
|---|---|---|
| Huấn luyện cơ chế | `SS-TRAIN`, `feature == base`, chưa bão hoà | tuyệt đối không có lưu lượng tính năng |
| Hiệu chỉnh ngưỡng u→SLO | ramp `base` (Phase 0) | không tính năng |
| Phát triển (dev) | ramp `promo`, `recs` | được dùng để chỉnh mô hình |
| **Kiểm tra KHÓA** | ramp `track`, `review` | không được đọc khi chỉnh mô hình; chỉ mở một lần khi đóng băng |
| Đối chứng | `review` (tác động nhỏ), chain sai ngẫu nhiên | kiểm tra không báo động giả / chain có giá trị |

## 7. Vệ sinh thí nghiệm (đã cài trong harness)

Index `customerId` trên `orders-db` (tránh chi phí tăng theo số đơn); pool tài khoản riêng (tránh giỏ hàng cộng dồn);
thứ tự lượt xáo ngẫu nhiên; front-end tạo lại trước mỗi pha; loadgen chạy trên host và tự báo khi bão hoà
(`generator_limited`); ghi `host_cpu`, `vm_cpu_util`; manifest ghi trước khi đo (pre-register) kèm SHA của mã nguồn.

## 8. Giới hạn đã biết (ghi sẵn cho Threats to Validity)

- **Trạng thái "không trần"** được biểu diễn bằng `--cpus = số vCPU của VM` (12), không phải NanoCpus rỗng: `docker update --cpus 0` là lệnh **không làm gì** (đã kiểm chứng), gỡ hẳn chỉ bằng tạo lại container. Harness đọc lại `docker inspect` sau mỗi lần áp trần và dừng nếu lệch; collector coi `NanoCpus ≥ NCPU` là không trần (`spec-cpu-quota = 0`).

- Bốn tính năng do người xây dựng viết sau khi đã đọc taxonomy ⇒ chuỗi dịch vụ khớp một phần là do cách cài, không phải bằng chứng độc lập; đáp án phụ thuộc cách cài (index/cache).
- Trần CPU là cấu hình nhân tạo (docker `--cpus`), hệ thống một máy; điểm gãy của ramp là điểm gãy **động** (bậc 45 s), chưa phải trạng thái ổn định. ⚠
- Loadgen đơn tiến trình đáng tin đến ~250 req/s.
- ⚠ Bốn giả định chưa kiểm: C1 đủ tạo điểm gãy trong tầm tải; JVM chạy ổn ở 0,15 core; SLO mặc định (p99 0,25 s, lỗi 1%) phù hợp; CPU/request không đổi theo loại request (thấy 0,65–1,25× ở backend, riêng front-end 1,3–2,1×, từ 5 mẫu/ô).
- Chưa có phép so với nhiều cách cài hoặc người cài độc lập.

## 9. Việc CHƯA chốt (cần người dùng quyết)

1. Ngưỡng SLO cuối cùng (chốt ở Phase 0).
2. Có thêm bản cài thứ hai của mỗi tính năng không (tăng thời gian ~2×). Cấu hình trần khác `RE2` đã bị loại (C1, C2 thất bại, xem 5c); nếu muốn nghiên cứu chế độ throttling thì cần mô hình riêng.
3. Có cho agent/người khác cài tính năng từ mô tả gốc (không thấy taxonomy) để đáp án độc lập không.
4. Có thêm tính năng loại khác (ghi nhiều, bất đồng bộ) không.
