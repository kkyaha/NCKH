# KHUNG CƠ CHẾ VÀ DỮ LIỆU — chốt trước khi thu thập

> Phạm vi: **chỉ** định nghĩa bài toán, hợp đồng dữ liệu và kế hoạch thu. **Không** tối ưu mô hình
> (làm sau khi có dữ liệu). Các mục đánh dấu ⚠ là giả định CHƯA kiểm chứng.
> Đọc kèm: `docs/HE_THONG.md` (bối cảnh), `experiments/collect/load_sweep_collect.py --help`.

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
mà bộ dự đoán sẽ gặp khi triển khai. Cưỡng chế bằng `experiments/collect/data_contract_check.py` (mục "tuân thủ schema chuẩn").
Đã kiểm chứng: **RE2-SS thật đạt 0 FAIL** (chỉ thiếu `routes.csv`), run dựng bằng harness đạt 0 FAIL / 0 WARN (offline).

Phát hiện khi đối chiếu: `metrics.csv` của RE2-SS **có giới hạn CPU** (`container-spec-cpu-quota`), nên nhận định cũ
("không dataset nào có K8s limit") **không đúng với RE2-SS**. Trần `C_s` là telemetry chuẩn đọc được
từ cAdvisor, không phải đầu vào tự chế.

## 5. Ma trận thu thập

Thư mục gốc **tách riêng** để loader không trộn dữ liệu có tính năng vào train.

| Pha | Mục đích | Lệnh (rút gọn) | Gốc dữ liệu | Ước tính |
|---|---|---|---|---|
| **0** hiệu chuẩn | xác nhận trần RE2, chốt SLO, kiểm tính bất biến của CPU/request khi có trần | `--ramp --limits RE2 --features base --repeats 3` | `SS-LIMITS` | ~40 ph |
| **A** huấn luyện | học cơ chế; KHÔNG tính năng, KHÔNG trần | `--levels 10,25,50,75,100,150,200,250 --repeats 2 --hold 240 --warmup 45` | `SS-TRAIN` | ~75 ph |
| — đóng băng | ghi dự đoán P0/P1/P1_ctrl (hash) từ Pha A + Phase 0 **trước** khi chạy B | `experiments/feasibility/freeze_predictions.py` | `data/processed/frozen/predictions_frozen_<cfg>.json` | vài giây |
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

## 5d. Kết quả Pha B trên tập PHÁT TRIỂN (2026-09-22; bản đóng băng RE2, SHA-256 `f05c8eb5…bc8e2`; baseline + promo + recs, 13 ramp)

Tập khoá (`track`, `review`) **chưa mở** (không có `locked_access.log`). Bản đóng băng không bị chỉnh sau khi thấy kết quả.

| Ô | Đo được [lo, hi) | P0 = P1 = ctrl | Sai số vs lo |
|---|---|---|---|
| baseline | [220, 240) | 207 (front-end) | −6% (an toàn) |
| promo ×1 | [150, 170) | 172 | +15% (**kha thi gia**) |
| promo ×2 | [80, 100) | 148 | +85% (**kha thi gia**) |
| recs ×1 | [120, 140) | 159 | +32% (**kha thi gia**) |
| recs ×2 | [100, 120) | P0/ctrl 129 (+29%); **P1 119 (đúng, nghẽn dự đoán ở orders)** | |

* Phân quyết trên lưới tải (96 điểm): P1 acc 0,906, 9 "khả thi giả"; P0 và ctrl acc 0,885, 11 khả thi giả; **0 báo động giả** ở cả ba.
  ⚠ Con số của `ctrl` cũng là một lần bốc: chạy lại 200 lần, `chain sai` có acc **trung vị 0,913 = đúng bằng chain thật**, khả thi giả **trung vị 9 = đúng bằng chain thật**; `seed=0` (acc 0,894, 11 khả thi giả) nằm ở **phân vị ~95**, tức lần bốc *bất lợi nhất* cho đối chứng. Ở ba split còn lại (khoá, độc lập, tiến cứu), acc và số khả thi giả của chain sai **giống hệt chain thật trong cả 200 lần bốc, không dao động một đơn vị nào**.
  ⇒ **Ở mức phán quyết hệ thống, đối chứng chain-sai không có sức phân biệt** — vì nút nghẽn luôn là gateway, mà gateway nằm trong *mọi* chain theo đúng định nghĩa. Chỉ phép so ở **mức node** mới có chút sức phân biệt (xem hiệu chỉnh dưới), và ngay cả ở đó 8/16 ô vẫn hoà.
* Mức sử dụng từng node (điểm % của trần, trước điểm gãy): gateway sai **9 điểm (promo) và 16 điểm (recs)**, giống nhau cho mọi bộ dự đoán (chi phí điều phối bị bỏ qua, front-end bị dự đoán thấp ~21%);
  node backend TRONG chain: P0 8,8/10,2, P1 8,5/9,5, chain sai 7,0/7,7 (promo/recs); node ngoài chain: P1 tốt nhất ở promo (0,63) nhưng kém P0 ở recs (3,8 so với 2,0).
  orders: P0 thấp −71%, **P1 cao +103%**, ctrl −77%.
  ⚠ **Mọi con số "chain sai" ở trên là MỘT lần bốc ngẫu nhiên** (`FP.wrong_chain` mặc định `seed=0`) — xem hiệu chỉnh ngay dưới.

**⚠ HIỆU CHỈNH (đối chứng chain-sai chạy lại dưới dạng phân phối, `experiments/feasibility/control_chain_distribution.py`, 200 lần bốc):**
Kết luận (3) bên dưới **đã sai và được đảo lại**. `seed=0` là một lần bốc bất thường có lợi cho đối chứng: gộp mọi split, nó rơi vào **phân vị 8** của phân phối (riêng split dev và tiến cứu: **phân vị 0**, tức lần bốc *thuận lợi nhất* cho chain sai trong 200 lần).
Khi lấy phân phối thay vì một lần bốc, trên 16 ô (tính năng × node trong chain):

| | sai số mức sử dụng (điểm % của trần) |
|---|---|
| chain THẬT | **5,53** |
| chain SAI | trung vị **6,56**, khoảng 5–95% [5,71 – 7,31] |
| chain SAI tại `seed=0` | 5,86 (phân vị 8) |

**98%** trong 200 lần bốc cho chain thật tốt hơn; đếm theo ô: chain thật thắng 6, chain sai thắng 2, **hoà 8/16**.
Nhưng hiệu ứng **nhỏ** (16% tương đối) và **8/16 ô không phân biệt được** — đối chứng yếu *do cấu tạo*: chain ngẫu nhiên cùng kích thước luôn chứa gateway nên trùng lặp nhiều với chain thật. Phát biểu đúng: *chain thật tốt hơn chain sai một cách nhất quán về hướng, nhưng chưa đạt ý nghĩa thống kê ở n = 16 (Wilcoxon p trung vị 0,19 qua các lần bốc, khoảng [0,008 – 1,000])*.
Bản đóng băng `predictions_frozen_RE2.json` (SHA-256 `f05c8eb5…bc8e2`) **giữ nguyên** — `P1_ctrl` trong đó vẫn là đối chứng tiền đăng ký hợp lệ; phân phối ở đây là lớp robustness hậu kiểm, bổ sung chứ không thay thế.

**Đọc kết quả:** (1) cơ chế đúng cho baseline; (2) với tính năng chưa từng có cả hai bộ đều lạc quan nguy hiểm ở 4/5 ô (15–85%), nguyên nhân chính là chi phí gateway;
(3) ~~giả thuyết "chain thật tốt hơn tỉ lệ nền và tốt hơn chain sai" **không được ủng hộ ở mức node**~~ → **xem hiệu chỉnh ở trên**: chain thật *có* tốt hơn chain sai (98% số lần bốc), hiệu ứng nhỏ và chưa đạt ý nghĩa; chain giúp rõ nhất ở ô mà orders là nút nghẽn thật (recs ×2).
Nguồn sai số nghi ngờ (chưa tách): độ lớn (k_s = 1 và chi phí mỗi lần gọi bằng trung bình nền, orders bị dự đoán gấp đôi) và chi phí gateway; cần phân tích P2 (bội số đo được từ `routes.csv`) trên dữ liệu phát triển.
Mọi chỉnh mô hình từ đây là **phát triển**, chỉ dùng promo/recs; tập khoá mở đúng một lần khi kết thúc.

## 5e. Kết quả trên tập KHOÁ (track, review) — mở đúng một lần (2026-09-22 06:19:09, `locked_access.log`)

Bốn mô hình chấm trên hai tính năng chưa từng dùng để chỉnh: P0/P1/chain-sai (đóng băng trước Pha B, SHA-256 `f05c8eb5…`) và **P2** (tham số fit trên promo/recs, đóng băng 06:18:49, SHA-256 `4c692d4f…`, trước khi mở tập khoá).

| Ô | Đo được [lo, hi) | P0 = P1 = ctrl | P2 |
|---|---|---|---|
| review ×1 | [180, 200) | 187,8 (+4%) | 176,0 (−2%) |
| review ×2 | [150, 170) | 172,2 (**+15%, muộn**) | 153,3 (+2%) |
| track ×1 | [170, 190) | 179,7 (+6%) | 163,9 (−4%) |
| track ×2 | [140, 160) | 158,9 (+14%) | 135,8 (−3%) |

* **P2 sai trong ±4% ở cả 4 ô, không ô nào dự đoán muộn** (0 "khả thi giả"); 3/4 nghiêng về an toàn. Trên lưới tải P2 có 6 báo động giả, tất cả nằm sát biên (bước lưới 20 req/s).
* Mức sử dụng node (điểm % của trần): track P0 3,9 / P1 4,8 / chain-sai 4,6 / **P2 1,9**; review 2,5 / 2,1 / 2,9 / **1,7**. ⚠ Cột `chain-sai` là một lần bốc (`seed=0`); trên 200 lần bốc, split khoá cho chain thật 5,09 so với chain sai trung vị 5,89 (khoảng 5–95% [4,74 – 7,20]), `seed=0` = 7,20 nằm ở **phân vị 89** — lần này lệch theo hướng *bất lợi* cho đối chứng. Xem `experiments/feasibility/control_chain_distribution.py`. Riêng node TRONG chain: track P1 13,3 → **P2 0,8**; review P1 1,0 → **0,6**. Gateway: 3,9–5,0 → 2,8–2,9.
* orders (sai tương đối): P0 −26%, P1 +41%, **P2 −2%**; front-end: −8,7% → **+0,8%**.
* Giả thuyết chưa kiểm ở dev, nay được kiểm: chi phí gateway theo số lời gọi backend (dev đều n_calls=3, khoá n_calls=2) **đứng vững**.
* Không giải thích được: `carts` bị dự đoán thấp ~22% dù không thuộc chain (cả P1 và P2), nghi do tác động gián tiếp; sai số tuyệt đối nhỏ.

**Giới hạn:** 2 tính năng khoá × 2 cường độ × 2 lần lặp; một máy, một hệ thống; các tính năng do người xây dựng viết (không độc lập); P0/P1 trông ổn hơn trên tập khoá vì hai tính năng này nhẹ hơn promo/recs.
Tập khoá đã dùng: mọi tính năng mới phải là dữ liệu mới.

## 5f. Dữ liệu ĐỘC LẬP (2026-09-22, đang thu): tính năng do một agent riêng cài

Mục đích: bỏ hai điểm yếu còn lại: (a) tập khoá `track`/`review` đã dùng, (b) bốn tính năng đầu do chính người xây dựng công cụ cài sau khi đã đọc taxonomy.

**Giao thức (thứ tự bắt buộc):**
1. Chọn 3 tính năng thuộc 3 archetype chưa dùng, có số lời gọi backend khác nhau: `cartsum` = VIEW_CART (n=1, REQ-06), `quickadd` = ADD_TO_CART (n=2, REQ-05), `express` = PLACE_ORDER (n=6, REQ-10).
   Dữ liệu cũ chỉ có n=2 và n=3; n=1 và n=6 là ngoại suy thật cho chi phí gateway `x·n_calls`.
2. **Đóng băng dự đoán TRƯỚC khi bất kỳ dòng code nào được viết:** `data/processed/frozen/predictions_frozen_RE2_P2_indep.json`,
   SHA-256 `f9853d23…3390`, tạo 06:34:45. Chỉ phụ thuộc mô tả tiếng Việt → archetype → chain, u\* và cơ chế của bản gốc, tham số P2 fit trên promo/recs.
3. Agent cài trong **thư mục cách ly** ngoài repo (`indep_sandbox`): chỉ có mã front-end gốc, danh mục API backend trung lập và yêu cầu (nguyên văn tiếng Việt + giao diện HTTP bắt buộc).
   Đã quét không có dấu vết taxonomy, chain, dự đoán, mã tính năng của người xây dựng hay bài báo. Agent không được chạy hệ thống.
4. Người xây dựng chỉ build, thử chức năng (`experiments/collect/probe_feature_chain.py`) và đo; lỗi chức năng được chuyển lại cho agent, KHÔNG chuyển gợi ý về thiết kế.
5. Đo ramp: baseline + 3 tính năng × 2 cường độ × 2 lần lặp dưới trần `RE2`. Đánh giá bằng `evaluate_frozen.py --split indep`, kèm so chuỗi gọi THỰC TẾ với chain taxonomy.

**Giao diện HTTP (bên yêu cầu cố định, backend nào bị gọi do người cài quyết định):** `GET /cart/summary`, `POST /cart/quick {id}`, `POST /checkout/express {id}`.
Điều này kiểm tra cả bản đồ tuyến đường (taxonomy có mô tả đúng một cài đặt tự nhiên không) lẫn độ lớn chi phí.

## 5g. Kết quả trên dữ liệu ĐỘC LẬP (2026-09-22) — phát hiện quan trọng: bội số gọi (k_s) không phải luôn ~1

16 ramp (`SS-LIMITS-INDEP`, 0 FAIL). Cả bốn bộ dự đoán đóng băng — kể cả **P2** (đã tổng quát hoá tốt trên tập khoá track/review) —
**sai rất lớn theo hướng lạc quan nguy hiểm**: điểm gãy dự đoán muộn hơn thực tế 14% đến **313%**.

| Ô | Đo được [lo, hi) | P0=P1=ctrl | P2 |
|---|---|---|---|
| cartsum ×1 | [160, 180) | 188 (+17%) | 182 (+14%) |
| cartsum ×2 | [140, 160) | 172 (+23%) | 162 (+16%) |
| quickadd ×1 | [60, 80) | 180 (**+200%**) | 164 (+173%) |
| quickadd ×2 | [40, 60) | 159 (**+297%**) | 136 (+240%) |
| express ×1 | [40, 50) | 165 (**+313%**) | 115 (+187%) |
| express ×2 | vỡ ngay từ bậc đầu (40) | 138 | 79 |

**Nguyên nhân đo trực tiếp** (`experiments/collect/probe_feature_chain.py`, độc lập với ramp): bội số gọi thật mỗi lần dùng tính năng

| Tính năng | Taxonomy giả định (k=1 mỗi node trong chain) | Đo được |
|---|---|---|
| cartsum | carts=1 (thiếu catalogue) | catalogue=1 (ngoài chain), carts=1 |
| quickadd | carts=1 | **carts=2** (có lời gọi đọc lại giỏ sau khi thêm) |
| express | user=1, carts=1 | **user=6**, **carts=3** |

Bốn tính năng chính (promo/recs/track/review) đều có k≈1 đo được, nên cả P1 lẫn P2 (chỉnh chi phí mỗi lần gọi, không chỉnh k) chưa từng bị thử với k≠1 trước đây — **đây là lần đầu**. `carts=2` của quickadd là lựa chọn cài đặt (đọc lại sau khi ghi). `user=6` của express nhiều khả năng một phần đến từ chính dịch vụ `orders` (tự phân giải các href HATEOAS của customer/address/card khi tạo đơn) — tức là hành vi **có sẵn trong hệ thống**, không phải lựa chọn của agent; archetype PLACE_ORDER chưa từng được dùng làm tính năng mới trong bốn tính năng chính nên hệ số này chưa từng lộ ra.

**Kết luận:** taxonomy (chain + k=1 mặc định) là xấp xỉ tốt cho các tính năng mà lối cài "tự nhiên" khớp với giả định một-lần-gọi-mỗi-hop
(đã đúng ở 6/6 tính năng trước). Nó **không đáng tin** khi lối cài thực tế gọi lại một backend nhiều lần — điều một công cụ dự đoán *trước khi cài*
không thể biết chắc. Đây là giới hạn cần nêu rõ trong Threats to Validity: công cụ cần một cơ chế phát hiện/cảnh báo k≠1 (ví dụ đo nhẹ ở tải thấp trước
khi đưa ra phán quyết cuối, giống `probe_feature_chain.py`), hoặc chấp nhận khoảng bất định rất rộng khi k chưa được xác nhận.

## 5h. P3 (k đo được thay k=1 giả định) — chẩn đoán hồi cứu, sửa được một phần, lộ thêm nguyên nhân thứ hai

**Không phải mô hình mới:** `FeasibilityPredictor.workloads(mode='P2', k=...)` đã nhận tham số `k` từ trước; `freeze_predictions.py` chỉ chưa từng truyền nó (mặc định k=1 mọi node). P3 = P2 (giữ nguyên `c_s`, `x` đã fit trên promo/recs) + `k` đo bằng probing chức năng nhẹ
(`experiments/collect/probe_feature_chain.py --save`, ~20 lời gọi/tính năng, KHÔNG dùng dữ liệu tải/điểm gãy). Sửa thêm một chỗ: công thức chi phí gateway trước đó dùng **độ dài chain** làm số lượt gọi backend;
đổi thành **Σk** (tổng bội số thật) — khi k mặc định thì Σk = độ dài chain, tương thích ngược hoàn toàn (18/18 test cũ vẫn qua).

⚠ **Đây là chẩn đoán HỒI CỨU**, không phải dự đoán đóng băng mới: dữ liệu độc lập đã được xem (`evaluate_frozen.py --split indep` chạy trước đó). Việc đo `k` tự nó không cần dữ liệu tải nên về nguyên tắc
làm được *trước* khi phán quyết, nhưng để tuyên bố P3 tổng quát hoá tốt cần một vòng dữ liệu độc lập **mới, chưa từng đo**, đóng băng P3 trước.

**Kết quả (sai số bình quân |%| so với ngưỡng đạt SLO thấp nhất, tập độc lập):**

| Mô hình | Tập độc lập | Tập dev (đối chứng) | Tập khoá (đối chứng, k~1 đúng) |
|---|---|---|---|
| P1 (k=1) | 170% | 38% | 10% |
| P2 (c,x; k=1, chain=độ dài) | 126% | 14% | 3% |
| **P3 (k đo được, gateway=Σk)** | **104%** | 15% (không đổi có ý nghĩa) | 3% (không đổi, đúng như kỳ vọng) |

P3 sửa mạnh nhất ở tính năng thiên về CPU: **express từ +313% xuống +111%**, cartsum từ +17%/+23% xuống +10%/+14% (do đã thêm cả biến thể "P3+chain" gộp catalogue vào VIEW_CART). Đối chứng đúng như kỳ vọng:
dev và tập khoá gần như không đổi (chúng vốn có k≈1 đo được).

**Nguyên nhân sai số còn lại (mới, đo trực tiếp, tách bạch với vấn đề k):** so mức sử dụng CPU thật tại đúng bậc tải nơi SLO vỡ —

| Tính năng | front-end lúc vỡ SLO | carts lúc vỡ SLO | Cơ chế |
|---|---|---|---|
| cartsum ×1 | ~84% (gần u\*=0,878) | — | **CPU bão hoà** — đúng cơ chế P0–P3 mô hình |
| express ×1 | 38–55% | 20–29% | SLO vỡ (p99 98→218ms) khi **chưa node nào gần trần** |
| quickadd ×1 | 47–49% | 34–36% | SLO vỡ (p99 209→2499ms) khi **chưa node nào gần trần** |

Với express và quickadd, độ trễ vỡ ngưỡng SLO (p99 ≤ 250ms) ở mức sử dụng CPU **thấp hơn nhiều** so với `u*` hiệu chỉnh từ baseline. Đây là hiệu ứng **hàng đợi/độ trễ đuôi** do tính năng gọi
nhiều lượt backend (song song hoặc tuần tự): độ trễ đầu-cuối cộng dồn qua nhiều hop tăng nhanh hơn mức sử dụng CPU của bất kỳ node đơn lẻ nào. Không mô hình nào trong P0–P3 theo dõi độ trễ
đầu-cuối — cả bốn chỉ so **mức sử dụng CPU với một ngưỡng duy nhất**. Đây khớp với hạn chế đã biết từ trước của dự án (latency không dự báo được bằng cơ chế tuyến tính/hàng đợi hiện có,
xem `docs/HE_THONG.md (muc 6)`), không phải lỗi mới — nhưng đây là **lần đầu nó ảnh hưởng trực tiếp đến phán quyết khả thi** thay vì chỉ ảnh hưởng độ chính xác dự báo một con số.

**Kết luận cho bài báo:** hai nguyên nhân sai số ĐỘC LẬP, cần hai hướng khắc phục khác nhau —
(1) bội số gọi k≠1 — sửa được bằng đo trước khi phán quyết (P3, đã kiểm chứng hồi cứu, cần dữ liệu mới để xác nhận tiến cứu);
(2) hiệu ứng hàng đợi/độ trễ đuôi từ quạt-ra nhiều backend — **chưa có hướng khắc phục** trong khung `u_s = CPU_s/C_s` hiện tại; cần một chỉ báo riêng cho tính năng có nhiều lượt gọi
(ví dụ ngưỡng cảnh báo khi Σk vượt một mức, độc lập với việc CPU có gần trần hay không) thay vì cố mô hình hoá độ trễ (đã thử và bỏ, xem mục 8).

## 5i. Đường bất đồng bộ (rabbitmq/queue-master) — không cần đo mới, dùng dữ liệu đã có

Collector ghi CPU của MỌI container (không chỉ 7 service chính), nên `rabbitmq`/`queue-master` đã có sẵn trong mọi ramp đã chạy (mọi tính năng đều đi qua `orders`→`shipping`→hàng đợi `shipping-task`→`queue-master`).
Kiểm tra trên toàn bộ dải tải đã đo (40–260 req/s, ramp express + baseline): `rabbitmq_cpu` giữ phẳng quanh 1,6–2,7% (max 3,7%), `queue-master_cpu` quanh 0,2–0,4% (max 0,95%), **không có xu hướng tăng**.
Kết luận: đường bất đồng bộ không phải nút nghẽn trong dải tải đã kiểm; hai service này không cần đưa vào `SCORED` hay trần CPU. Không cần một tính năng "bất đồng bộ" riêng để kiểm việc này.

## 5j. Tích hợp vào hệ thống, khoảng bất định, trần từ telemetry (2026-09-22)

**`src/agents/feasibility_agent.py` (`NewFeatureFeasibilityAgent`), TÁCH RIÊNG khỏi `CapacityAgent`.** `CapacityAgent.train()` học từ dữ liệu RCAEval fault-injection, phục vụ RQ1–RQ12 đã công bố —
KHÔNG đụng vào để không rủi ro tái hiện các RQ đó. Agent mới trả lời một câu hỏi khác ("thêm tính năng CHƯA TỪNG CÓ thì hệ đang chạy có còn đáp ứng SLO ở tải đỉnh L không"), học từ `SS-TRAIN`/`SS-LIMITS`.

* **Trần `C_s` đọc TRỰC TIẾP từ container đang chạy** (`docker inspect …NanoCpus`) tại thời điểm hỏi, không phải từ `limits.json` tĩnh — chịu được lệch cấu hình. Có phương án dự phòng về `limits.json` kèm cảnh báo rõ nếu Docker không gọi được (ví dụ môi trường CI).
* **Khoảng bất định cho điểm gãy:** bootstrap KHÔNG tham số trên chính các hàng dữ liệu huấn luyện (không phải trên tham số đã fit) — resample có hoàn lại, fit lại cơ chế (rho/alpha/beta) mỗi lần, tính lại điểm gãy, lấy phân vị [5,95] thực nghiệm. 200 lần lặp mất ~2,3 giây.
* **Tổng quát hoá sang MỌI archetype** (không chỉ 8 tính năng có bằng chứng thực nghiệm): `feasibility_predictor.spec()` nhận cả tên tính năng đã đặt (`'promo'`) lẫn tên archetype thật (`'LOGIN'`, `'REGISTER'`…) trực tiếp.
* **Cờ `latency_risk`:** khi Σk (tổng bội số gọi backend) ≥ 4, verdict vẫn tính theo CPU nhưng kèm cảnh báo rõ — vì hiệu ứng hàng đợi/độ trễ đuôi (mục 5h) chưa được mô hình hoá; **không giả vờ đã giải quyết**, chỉ tránh im lặng bỏ qua.
* **Nối với `ParserAgent`:** `orchestrator.assess_new_feature_requirement(text, L_peak)` — đường RIÊNG, không đụng `StateGraph` `feasibility_analyzer` cũ. Đã kiểm chứng đầu-cuối (né import `capacity_agent`/`dowhy` bị lỗi môi trường có sẵn, xem dưới): NL → `APPLY_PROMO_CODE` (delta 20%) → verdict INFEASIBLE tại L=150, khớp chính xác kết quả `evaluate_p3.py` (145,4 req/s); thử với `LOGIN` (chưa từng có bằng chứng thực nghiệm) → FEASIBLE, `extrapolating=False`, hợp lý.
* **30 test mới** (`tests/test_feasibility_agent.py`), tất cả qua.

⚠ **Phát hiện phụ, KHÔNG do phiên này gây ra:** `tests/test_capacity_agent.py` đã hỏng từ trước (`ModuleNotFoundError: dowhy.graph`, do `.venv` có `include-system-site-packages=true` và `dowhy` hệ thống ở bản 0.8 thiếu submodule `graph`). KHÔNG nâng cấp `dowhy` vì việc đó sửa đổi Python hệ thống dùng chung, ngoài phạm vi được uỷ quyền — cần người dùng quyết định và tự làm.

## 5k. Điều tra nguyên nhân gốc của độ bất ổn đo lường — giả thuyết NAT bị bác bỏ

Giữa chừng đo `SS-LIMITS-PROSP`, điểm gãy baseline trôi liên tục trong một phiên chạy dài (xem mục 5j và `sockshop-data-collection-setup` — 220–240 → 180–220 → 140–160 req/s). Giả thuyết đầu tiên: lớp NAT/port-forward của Docker Desktop trên Windows (đo được p90 thời gian kết nối nhảy từ 1,4ms lúc rảnh lên 17,8ms dưới tải ~150 req/s) đang gây nhiễu có hệ thống vào phép đo.

**Thử để sửa (bị bác bỏ bằng thực nghiệm):** đóng gói loadgen (`load_sweep_collect.py`) vào một container gắn THẲNG vào `sockshop_default`, gọi service qua tên DNS nội bộ (`http://edge-router`) thay vì `127.0.0.1`, để bỏ qua lớp NAT. Kiểm chứng trực tiếp — cùng mức tải 80 req/s, cùng trần CPU RE2, cùng thời điểm:

| Nguồn tải | p99 | Lỗi | SLO |
|---|---|---|---|
| Host (qua NAT) | 67 ms | 0% | Đạt |
| Container (gắn thẳng mạng docker) | **29 212 ms** | 0% | **Vi phạm nặng** |

Container hoá loadgen — vốn định sửa lỗi NAT — **tự nó gây ra một lỗi giả còn nghiêm trọng hơn NAT rất nhiều** (29 giây so với 67ms, ở đúng một mức tải mà host xử lý hoàn toàn bình thường). Đã loại trừ hai giả thuyết cho lỗi container: (a) không phải CPU throttling — `docker stats` toàn bộ 14 container dịch vụ trong lúc "kẹt" đều dưới 30%; (b) không phải bản thân `LoadGen.setup()` chậm — gọi trực tiếp hàm thật (không qua bản sao chép tay) trong cùng container chỉ mất 24 giây cho 300 tài khoản. Nguyên nhân cụ thể bên trong container chưa được xác định (không đào sâu thêm — quyết định người dùng: xem giá trị chi phí đầu tư không rõ ràng so với việc quay lại đo từ host).

**Kết luận:** lớp NAT của Docker Desktop là có thật và đo được, nhưng độ lớn (~17ms) quá nhỏ so với ngưỡng SLO (250ms) để là nguyên nhân của các đợt trôi điểm gãy quan sát được — đó KHÔNG phải nguyên nhân gốc. Nguyên nhân trôi thực sự nhiều khả năng vẫn là tranh chấp tài nguyên host/VM tích luỹ sau phiên chạy dài (đã chẩn đoán ở mục 5j: restart Docker Desktop + `wsl --shutdown` phục hồi hoàn toàn điểm gãy về 240/260, tốt hơn cả mức gốc). Đã BỎ hướng container hoá harness: xoá `deploy/sockshop/harness-runner/`, revert phần đường dẫn `_host_path()`/`SS_HOST_ROOT` trong `load_sweep_collect.py` (giữ `SS_TARGET` như một override chung, không còn dùng để né NAT). Ghi vào Threats to Validity: độ trễ NAT ~17ms dưới tải là một giới hạn nhỏ, đã định lượng, của việc đo trên Windows/Docker Desktop thay vì Linux gốc.

Chiến dịch đo sạch `SS-LIMITS-CLEAN` (baseline + cartsum + quickadd + express + browse, 2 cường độ × 2 lần lặp, trần RE2) được chạy LẠI TỪ ĐẦU trên host sau khi sửa xong, thay thế mọi số liệu độc lập/tiến cứu trước đó.

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
