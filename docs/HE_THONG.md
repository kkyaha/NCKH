# Hệ thống — tài liệu tổng hợp

> **Đây là tài liệu tiếng Việt DUY NHẤT mô tả hệ thống hiện tại.** Mọi tài liệu tiếng Việt
> khác trước 2026‑09‑24 đã bị xoá vì dùng hệ đánh số RQ cũ và mô tả các phiên bản đã bị thay
> thế. Hai file còn lại có vai trò riêng:
> - `docs/paper_draft.tex` — **nguồn số liệu chính thức duy nhất**. Mâu thuẫn với tài liệu
>   này thì paper đúng.
> - `docs/DATA_FRAMEWORK.md` — nhật ký thực nghiệm chi tiết của chiến dịch đo `SS-*`
>   (Phase 0/A/B, sự cố trôi hiệu năng, các hiệu chỉnh). Giữ nguyên làm hồ sơ xuất xứ.

---

## 1. Bài toán và phạm vi

**Câu hỏi hệ thống trả lời:** cho một *yêu cầu tính năng chưa tồn tại*, viết bằng ngôn ngữ tự
nhiên — hệ đang chạy còn đáp ứng SLO ở tải đỉnh kỳ vọng không, và node nào nghẽn trước?

Đây **không phải** dự báo chuỗi thời gian. Đây là **suy luận can thiệp** (*what‑if*) trên một
hệ chưa có tính năng đó.

### Phạm vi — đọc kỹ, đây là ranh giới cứng

Hệ thống nhận yêu cầu **thêm lưu lượng lên kiến trúc sẵn có**: tính năng mới được phục vụ bởi
các service **đã triển khai và đã có telemetry**, đi theo các cạnh gọi **đã phát sinh lưu lượng
nền**.

Yêu cầu cần **service mới** hoặc **cạnh gọi mới** nằm **ngoài miền xác định** của mô hình — không
phải "chưa kiểm", mà là không có đại lượng để tính:

| Đại lượng | Vì sao node mới làm nó vô nghĩa |
|---|---|
| `(α_s, β_s)` cơ chế CPU | fit từ lịch sử telemetry của **chính service đó** |
| `ρ_s` tỉ lệ workload | fit từ đồng biến thiên với gateway trên lưu lượng nền |
| `C_s` trần CPU | đọc từ hạn ngạch của **container đang chạy** |
| hệ số Tier‑1 của cạnh | fit từ đồng biến thiên của cạnh đó trên lưu lượng nền |

Trả lời được câu hỏi đó đòi **khám phá cấu trúc nhân quả**, còn bài này đứng trên giả định
*đồ thị đã biết* — hai bài toán khác nhau. Cả 8 tính năng đã đo đều tuân thủ ranh giới này.

⚠ **Lỗ hổng chưa xử lý:** scope gate từ chối yêu cầu không khớp archetype nào **về ngữ nghĩa**,
nhưng **không có gì phát hiện được** yêu cầu cần service mới **về kiến trúc**. Một yêu cầu như
vậy có thể khớp một archetype sẵn có rồi nhận phán quyết tự tin tính trên tập service sai.

---

## 2. Kiến trúc: hai hệ tách biệt, có chủ đích

Repo chứa **hai bộ dự đoán khác nhau**. Nhầm lẫn giữa chúng là nguồn hiểu sai lớn nhất.

### Hệ A — `CapacityAgent`: DAG nhân quả 2 tầng, 28 node

Dùng cho RQ1–RQ4 (dữ liệu RCAEval / Alibaba, fault‑injection).

- **Node:** 7 service × 4 metric (`workload`, `cpu`, `mem`, `latency-50`) = 28
- **Cạnh cấu trúc (29):** Tier‑1 `workload→workload` theo 8 cạnh gọi thật; Tier‑2
  `workload→{cpu, mem, latency}` trong cùng service
- **Cạnh phải học (3/8 ứng viên sống sót)** qua held‑out 67/33 → knee‑point → OOD‑safety:
  `orders_cpu→shipping_cpu`, `orders_cpu→carts_cpu`, `front-end_cpu→user_cpu`.
  **0 cạnh** latency‑backprop qua được cổng.
- **Cơ chế:** `*_cpu`/`*_mem`/`*_workload` dùng `LinearRegression(positive=True)`;
  `*_latency-50` dùng `QueueingLatencyRegressor` (hồi quy trên `[X, X/(C−X)]`, `C = P99(X)×1.5`)

### Hệ B — `FeasibilityPredictor`: hai tầng phẳng

Dùng cho RQ5 (testbed `SS-*` tự dựng). **Không dùng DAG của hệ A.**

```
ρ_s   : hồi quy QUA GỐC của W_s theo W_gateway
CPU_s = α_s + β_s · W_s          (NNLS, ràng buộc α,β ≥ 0)
```

**Không có cạnh Tier‑1 nào.** Mọi service treo thẳng vào gateway qua `ρ_s`. Chain chỉ là
**mặt nạ cộng tải**, không phải đường lan truyền. **Không có node `latency` nào** — đây là lý
do cấu trúc khiến tiêu chí khả thi buộc phải thuần CPU.

**Vì sao không hợp nhất hai hệ?** Đã đo: mượn Tier‑1 của hệ A cho hệ B làm **tệ đi**
(30,9% → 36,7%), vì taxonomy biểu diễn chain là *tập* nên hệ số cạnh bắn nhầm. Chi tiết ở §6.

---

## 3. Thang bóc tách P0 → P3

Các `P` **không phải mô hình cạnh tranh** — chúng là một **thang bóc tách**, mỗi bậc thêm
**đúng một** giả định, nên hiệu số giữa hai bậc quy được sai số về một nguyên nhân.

| | Công thức tải | Thêm gì | Tư cách bằng chứng |
|---|---|---|---|
| **P0** | `W_s = ρ_s·L·(1+Δ)` | — (baseline **B4**: rải đều theo tỉ lệ nền) | tiền đăng ký |
| **P1** | `W_s = L·(ρ_s + Δ·k_s·[s∈chain])` | **chain làm mặt nạ**, `k_s`=1 | tiền đăng ký |
| **P1_ctrl** | như P1, chain **ngẫu nhiên** cùng cỡ | đối chứng | tiền đăng ký |
| **P2** | backend `L·(ρ_s+Δ·k_s·c_s)`; gateway `L·(ρ+Δ(1+x·Σk))` | **chi phí mỗi lượt gọi** | tiền đăng ký |
| **P3** | = P2 với `k_s` **đo bằng probe** | **bội số gọi thật** | hồi cứu |

Phán quyết: `u_s = CPU_s/C_s` trên 5 node, ngưỡng `u* = 0,8783` hiệu chỉnh từ điểm gãy của
ramp baseline. `R*` có nghiệm đóng vì mọi thứ tuyến tính theo `L`.

Bản đóng băng: `predictions_frozen_RE2.json` (SHA‑256 `f05c8eb5…bc8e2`) chứa P0/P1/P1_ctrl;
P2 đóng băng riêng (`4c692d4f…`). **Không bao giờ tái sinh các file này** — chúng là chứng cứ
tiền đăng ký; tái sinh là phá dấu vết kiểm toán.

### Hai cổng bổ sung (đã hiện thực, **mặc định TẮT**)

Bật bằng cách truyền `feature_demand` + `slo_p99` cho `FeasibilityPredictor`.

| Cổng | Công thức | Hiệu quả đo được (tập độc lập) |
|---|---|---|
| **Nhu cầu phục vụ** | `Σ_s (k_s/Σk)·D_feat/(1−u_s(L)) ≤ SLO_p99` | 72,6% → **55,8%**; độ muộn nguy hiểm 405 → **243** |
| **Throttling** | `max_s g_s(u_s(L)) ≤ τ*`, `g_s` isotonic trên baseline | → **40,1%**; độ muộn → **143** |

`R* = min(R*_cpu, R*_demand, R*_throttle)`. Cả hai **vẫn là hồi cứu** — `D_feat` hiện lấy xấp xỉ
từ bậc tải thấp nhất của ramp. Chúng chỉ thành tiến cứu khi `probe_feature_chain.py` chạy trên
hệ thật và `D_feat` được đóng băng **trước** ramp (công cụ đã sẵn sàng, xem §8).

---

## 4. Dữ liệu — bộ nào dùng cho việc gì

| Bộ | Loại | Tên service thật | Biên độ tải | Dùng cho |
|---|---|---|---|---|
| RCAEval (RE2‑SS, RE2‑OB) | benchmark RCA | có | **không** | RQ1–RQ2, RQ4b |
| Alibaba v2021 | trace production 12h | không | có | RQ3, RQ4a |
| **`SS-*` (tự dựng)** | testbed có kiểm soát | có | có | **RQ5** |

### Các chiến dịch `SS-*`

| Thư mục | Vai trò |
|---|---|
| `SS-TRAIN` | học cơ chế — **không** tính năng, **không** trần CPU |
| `SS-LIMITS` | Phase 0 (hiệu chỉnh `u*`) + Pha B (dev: promo/recs; khoá: track/review) |
| `SS-LIMITS-CLEAN` | chạy lại sạch: độc lập (cartsum/quickadd/express) + tiến cứu (browse) |
| `SS-LIMITS-INDEP`, `SS-LIMITS-PROSP` | **đã bị thay thế** bởi CLEAN sau sự cố trôi hiệu năng |
| `SS-LIMITS-C1`, `SS-LIMITS-C2` | cấu hình trần đã **thất bại** vì CFS throttling (xem §7) |
| `SS-CALIBRATION`, `SS-LOADSWEEP` | tiền‑schema, thăm dò ban đầu — không dùng |

⚠ `data/raw/` nằm trong `.gitignore`. Dữ liệu **không theo git**; phải chuyển tay giữa các máy.
Bài lấy testbed làm đóng góp chính nên **bắt buộc phát hành bộ dữ liệu** khi nộp.

---

## 5. Bộ 5 RQ hiện tại

| | RQ | Mệnh đề | Kết cục |
|---|---|---|---|
| 1 | **Necessity** | năng lực generator làm verifier thành thừa | **bác** — Anchor MAE giảm 25× nhưng GMR vẫn 92% |
| 2 | **Guarantee** | (a) α chứng nhận tỉ lệ trượt; (b) hiện thực thoả đặc tả | **bác cả hai** |
| 3 | **Measurability** | dữ liệu đỡ được tuyên bố dự báo | **bác** — benchmark đảo ngược thứ hạng mô hình |
| 4 | **Attribution** | graph / do‑operator mang nội dung nhân quả | graph **sống**; do‑operator **bác** |
| 5 | **Prospective** | phán quyết khả thi đúng khi có ground truth | C6a **sống**; C6b **bác**; C6c bác một phần |

**Hai claim sống sót**, ba bị bác. Ba trong số các phản bác nhắm vào kết quả **chính nhóm đã
công bố trước đó**.

### ⚠ Bảng dịch — repo có BA thế hệ đánh số RQ

| Bài hiện tại | Khung V3 (đã xoá) | Hệ cũ RQ1–RQ12 (tên file `rq*_.py`) |
|---|---|---|
| RQ1 Necessity | RQ1 | RQ3 (benchmark parser 50 prompt) |
| RQ2 Guarantee | RQ2 + RQ3 *(gộp)* | — |
| RQ3 Measurability | RQ4 | RQ1, RQ2 |
| RQ4 Attribution | RQ5 | RQ4, RQ7–RQ11 |
| RQ5 Prospective | *(chưa có)* | RQ12 |
| — | — | RQ6 → tách sang `xai_attribution_paper_draft.tex` |

Nhãn LaTeX `sec:rq2…sec:rq6` **giữ tên cũ có chủ đích** (để 67 tham chiếu chéo không gãy) nên
**số trong nhãn lệch số hiển thị**; mỗi chỗ lệch có comment `% NOTE:`.

**Khi viết script mới: đừng đặt tên theo số RQ.** Đặt theo cơ chế nó đo.

---

## 6. Kết quả chính

### Cơ chế cấu trúc thắng ML hộp đen — vì bài toán là **ngoại suy**

Trên `SS-TRAIN`, giao thức OOD (train tải thấp 67% → test tải cao 33%, ngoại suy 2,05×):

| Mô hình | MAPE | skill | thắng/7 service |
|---|---|---|---|
| **NNLS (cơ chế triển khai)** | **4,87%** | **0,990** | **7/7** |
| GradBoost | 31,23% | 0,705 | 0 |
| GaussianProcess | 26,30% | 0,705 | 0 |
| Hằng số | 70,06% | 0,000 | 0 |

**Cơ chế** — ở 1,9× dải huấn luyện: sự thật **54,7**, NNLS **58,7**, GradBoost **đóng băng ở
30,7** (cây là hằng số từng mảnh, ngoài dải trả về giá trị lá ở biên), GaussProc **đảo dấu
xuống 15,5** (hồi về prior → dự đoán tải tăng thì CPU giảm). Xem `docs/figures/fig_extrapolation.pdf`.

⚠ **Sau hiệu chỉnh Holm, không so sánh cặp nào đạt p < 0,05** — với n=7 và m=5, sàn của
Wilcoxon là 0,0156 × 5 = **0,0781**, tức *không thể* đạt. Nhưng **effect size ở cực đại**:
rank‑biserial và Cliff δ đều **−1,000** (chi phối hoàn toàn). Phải báo cáo **omnibus + effect
size**, không báo p pairwise.

### Benchmark đảo ngược thứ hạng

| | Cliff δ (SCM vs GradBoost) |
|---|---|
| RCAEval | **+0,224** (yếu, mơ hồ — và **lật theo cỡ mẫu**: p = 0,11 vs 0,81) |
| SS‑TRAIN | **−1,000** (chi phối hoàn toàn) |

Trên RCAEval thứ hạng bị quyết định bởi **bậc tự do của người nghiên cứu**, không phải bởi dữ
liệu — đúng định nghĩa *vacuous*.

### Memory là chỉ số vô hiệu — đối chứng có kiểm soát

Cùng hàng dữ liệu, cùng giao thức, cùng lớp mô hình: CPU skill **+0,990** vs Memory
**0,000** (âm ở 2/7 service). CV của memory **0,001–0,016**; `carts_mem` dịch chuyển **0,5%**
toàn dải tải. Không còn đường đổ lỗi cho dữ liệu. Claim memory **đã rút** khỏi bài.

### Khả thi: nội dung nhân quả nằm ở **trọng số**, không ở topology

Sai số mức sử dụng node trong chain (điểm % của trần):

```
P0 uniform  7,49      P1 chain thật  6,46   ← topology mua được 1,03
P1 chain sai 7,02      P3 chain+k     1,46   ← TRỌNG SỐ mua được 5,00
```

Chain thật vs chain sai: **không tách được** (Wilcoxon p trung vị 0,19 qua 200 lần bốc,
**8/16 ô hoà**). Ở mức phán quyết, đối chứng chain‑sai có sức phân biệt **bằng không** — nút
nghẽn luôn là gateway, mà gateway nằm trong *mọi* chain theo định nghĩa.

### Điểm gãy: sai số kèm khoảng tin cậy bootstrap

| Tập | n | P1 | P2 | P3 |
|---|---|---|---|---|
| dev | 4 | 37,6% | 14,4% | 15,3% |
| **khoá** | 4 | 9,6% [5,0–14,2] | **2,7% [2,2–3,3]** | 2,7% |
| độc lập | 5 | 119,7% | 87,5% | 72,6% **[15,9–151,9]** |
| tiến cứu | 2 | 10,0% | 14,0% | *(n≤2, không báo khoảng)* |

Kết quả **tập khoá là chắc chắn nhất**. Tập độc lập có khoảng **cực rộng** — 72,6% không nên
trích dẫn đơn lẻ.

### Ba khe hở biểu diễn của taxonomy

| | Taxonomy không mang | Hậu quả |
|---|---|---|
| 1 | chain là **tập**, không phải **đường đi** | Tier‑1 bắn nhầm (`recs` user 7,46 vs 1,0) |
| 2 | không mang **bội số** `k_s` | `express` +313% |
| 3 | không mang **kiểu lời gọi** (đọc/ghi) | `quickadd` +96% |

Cả ba **đo được bằng đúng một công cụ** — probe đã thực hiện sẵn 20 lời gọi thật, chỉ cần ghi
thêm method, thời gian, số lượt. **Đơn thuốc: taxonomy phải là đường đi có trọng số và có kiểu.**

Hệ số Tier‑1 trên telemetry nền khôi phục đúng bội số thật ở nơi archetype **thực sự đi qua
cạnh** (`express`: carts 3,17 vs 3,0 thật; payment/shipping/orders/catalogue **1,00 tuyệt
đối**). Ở 4 ô có k>1,5: MAE giảm **2,25 → 0,91**. Nhưng nó đúng **2/4** — trượt đúng những ô
do *lựa chọn người cài* (`quickadd` đọc‑lại‑sau‑ghi, `recs` gọi catalogue hai lần).

---

## 6b. Giải trình phán quyết — `explain()` / `explain_text()`

`verdict()` trả về **kết luận**; `explain()` trả về **suy luận**. Một kỹ sư đọc
`INFEASIBLE, max_u=0.93` không biết con số đó từ đâu ra. `explain_text()` phơi bày từng bước
của chính `workloads()`/`cpu()`/`verdict()` — **không tính lại gì**, nên không thể lệch:

```
Yeu cau 'express' o tai nen L = 120 req/s  (che do P2)
  Delta = 0.250      chain = [front-end, user, catalogue, carts, orders, payment, shipping]
  boi so goi k: do bang probe

  service        W nen  +tinh nang      = W     CPU  / tran     = u
  front-end      120.0       173.7    293.7   61.95      50   1.239 !
  user            39.1       236.0    275.1   18.58      20   0.929  *
  carts           47.4       136.1    183.5   32.24      50   0.645  *

  ! nut nghen = front-end voi u = 1.239; nguong u* = 0.878
  => PHAN QUYET: INFEASIBLE
  * CANH BAO ngoai suy: [front-end, user, carts, orders] vuot dai tai da quan sat
  cong do tre: R_feat = 4358 ms vs SLO 250 ms -> VUOT
     · user: trong chain: Delta=0.250 x k=6 x c_s=1.311
```

Mỗi dòng **tự kiểm được bằng tay**: `W nền + W tính năng = W tổng`, `CPU = α + β·W`,
`u = CPU/trần` — có test cưỡng chế cả ba đẳng thức. Bảng cũng nói rõ **nguồn của `k`**
(`đo bằng probe` vs `mặc định 1 — CHƯA ĐO, bất định`), vì phán quyết phụ thuộc vào đó.

## 7. Bẫy đã biết — đọc trước khi sửa bất cứ thứ gì

1. **Gateway bị loại âm thầm khỏi DAG.** Bảng metrics của Online Boutique gọi gateway là
   `frontend`, graph (trích từ trace) gọi `frontendservice`. Hệ vẫn chạy trọn và in
   `[OK] Đã khớp Global DAG 24 nodes` trong khi **thiếu hẳn điểm vào**. Đã thêm chốt cảnh báo
   — nhưng **kiểm tên service giữa graph và metrics mỗi khi thêm hệ mới**.

2. **`extract_graph_generic.py` gán nhầm service thành `database`.** Heuristic
   "sink out‑degree=0 ⇒ database" đúng với SockShop/Train Ticket nhưng **sai với OB**: trace
   không ghi lời gọi datastore nên 4 service ứng dụng thật bị gán nhầm → `bfs_closure()` loại
   chúng khỏi mọi archetype. Đã sửa tay trong JSON.

3. **`compute_headroom` từng so trung bình với P99.** Hai họ thống kê khác nhau → phân phối
   lệch phải nặng làm trung bình vượt P99 → biên âm → CRITICAL giả. Đã sửa thành khoảng
   **P50→P99**. **Đừng quay lại dùng mean làm mốc dưới.**

4. **MAPE thấp không có nghĩa là mô hình tốt.** Target gần bất biến cho MAPE thấp dưới *mọi*
   predictor. Luôn dùng **skill score + negative control**.

5. **Đừng sửa** `evaluation_suite.py`, `select_scm_edges.py`,
   `backpressure_edge_ood_safety_test.py` — bản sao đông cứng có chủ đích, sinh số liệu đã công bố.

6. **Đối chứng ngẫu nhiên một‑lần‑bốc là bẫy.** `FP.wrong_chain()` mặc định `seed=0`; p của
   Wilcoxon trải **0,008 → 1,000** tuỳ hạt giống, và `seed=0` rơi vào **phân vị 8**. Một kết
   luận trong `DATA_FRAMEWORK.md` §5d đã **bị đảo ngược** khi chạy lại dạng phân phối. Dùng
   `wrong_chain_draws()` và luôn báo cáo phân phối.

7. **CFS throttling ở hạn ngạch nhỏ.** Với trần < 0,5 core, độ trễ bị chi phối bởi **bị bóp
   theo chu kỳ**, không phải mức sử dụng trung bình. Cấu hình C1 (carts 0,15 core) dùng **95%
   trần ở 40 req/s** trong khi mô hình tuyến tính dự đoán 19%. C2 (catalogue 0,08 core) vỡ SLO
   ở 140 req/s thay vì ~223 dự đoán. **Cả hai cấu hình đã bị loại.**

---

## 8. Hướng đã LOẠI — đừng làm lại

- **Chuẩn hoá theo instance (autoscaling).** `corr(workload, n_inst)` trung vị **0,0047**;
  chuẩn hoá làm **xấu đi** (R² 0,192 → 0,053), `raw_total` thắng ở 82,5% service, p≈6e‑18.
- **Mine call‑chain từ log cho Train Ticket.** `logs.parquet` là log nghiệp vụ Spring Boot;
  **0/197.087 dòng** khớp regex access‑log. Dùng BFS‑from‑seed. *(Với SockShop thì mine được,
  Jaccard 0,60 — nhưng không đồng đều nên **không** thay mặc định.)*
- **Dạng hàm log‑log elasticity.** Không có tiền lệ cho microservice và **xung đột với nền
  Kingman/queueing** đã trích trong bài.
- **Lấy trần `C` từ K8s resource limit.** Không bộ nào trong 4 bộ có → không đánh giá được.
- **`CEILING_TAIL_RATIO_MAX = 10.0`.** Phân bố `max/P99` liên tục (P50=1,67, P90=5,2,
  P95=26,3) — **không có vách ngăn tự nhiên tại 10**. Thay bằng bootstrap + `UNDECIDED`.
- **Hàng đợi M/M/1 mỗi trạm cho hệ B.** Fit hỏng: R² **âm** ở backend vì chúng chưa bao giờ
  tiến gần bão hoà (u ∈ [0,02, 0,35]).
- **Nút nghẽn ẩn (DB, socket).** Chấm lại **mọi** container có trần: nút nghẽn **luôn là
  `front-end`** ở cả 16 ramp; DB cao nhất 0,59.
- **Tranh chấp khoá ở `quickadd`.** Ở cùng mức sử dụng `carts`, độ trễ chỉ **0,93×** của
  `cartsum`. Nguyên nhân thật là **bất đối xứng chi phí ghi/đọc** (disk I/O `carts-db`
  **13,9×**, CPU mỗi lượt gọi **1,83×**).
- **Hợp nhất Tier‑1 của hệ A vào hệ B.** Làm **tệ đi** 30,9% → 36,7%.

---

## 9. Giới hạn đã biết

| | Giới hạn |
|---|---|
| **n nhỏ ở mọi nhánh** | tiến cứu **n=2**; khoá n=4; độc lập n=5; 7 service |
| **Một hệ, một máy, một topology** | gateway luôn nghẽn → không kiểm được "node nào nghẽn" |
| **Cơ chế fit trên dữ liệu KHÔNG trần, áp dụng DƯỚI trần** | bất biến ≤7% cho 5 service; `orders` 0,89× và `shipping` 0,49× chưa so sạch |
| **Ngưỡng SLO chọn sau khi xem dữ liệu nhạy cảm** | bậc tự do người nghiên cứu, đã ghi nhận |
| **Lưới tải 20 req/s tuyệt đối** | độ phân giải tương đối 10% ở điểm gãy 200 nhưng **50%** ở điểm gãy 40 → **sai số tương đối không so sánh được giữa các ô** |
| **`quickadd` còn +82% chưa giải thích được** | đã loại 3 giả thuyết, còn phần dư |
| **Cổng throttling chỉ tồn tại khi có hạn ngạch CPU** | triển khai không đặt `limits` thì không có tín hiệu |
| **Tính năng do agent viết ≠ do người viết** | độc lập với taxonomy đã chứng minh; **tính đại diện thì chưa** |

### ⚠ Một con số trong Abstract chưa tái lập được

`tab:signal-gate` ghi RCAEval R² = **0,0034** cho cả CPU lẫn Memory. Con số này là **hằng số
viết cứng** ở `alibaba_signal_check.py:177` — **không script nào trong repo tính ra nó**.
Tính lại từ dữ liệu: CPU **0,0150**, Memory **0,0010**, Socket **0,0204** (gộp rồi trung vị
theo service) hoặc 0,0092 / 0,0041 / 0,0064 (trung vị theo từng service × run). Biên độ tải
1,70 thì khớp (1,73).

**Kết luận của bài không đổi** (mọi giá trị ≤ 0,02 « 0,3009 của Alibaba) nhưng **con số phải
được làm lại có nguồn gốc trước khi nộp**.

---

## 10. Cách chạy lại

```bash
source .venv/bin/activate

# --- kiểm tra hợp đồng dữ liệu trước khi tin bất cứ số nào
python experiments/collect/data_contract_check.py --dir data/raw/SS-TRAIN       --role train
python experiments/collect/data_contract_check.py --dir data/raw/SS-LIMITS-CLEAN --role limits

# --- nhánh khả thi (RQ5). THỨ TỰ QUAN TRỌNG: freeze phải chạy TRƯỚC khi đo ramp
python experiments/feasibility/freeze_predictions.py ...        # đóng băng + hash
python experiments/feasibility/evaluate_frozen.py --frozen data/processed/frozen/predictions_frozen_RE2.json \
       --ramp-dir data/raw/SS-LIMITS --split dev
python experiments/feasibility/evaluate_p3.py                   # thang P1→P3 đầy đủ

# --- thống kê chặt: Friedman → Wilcoxon → Holm + effect size + bootstrap CI
python experiments/model_eval/statistical_rigor.py

# --- đối chứng chain-sai dạng PHÂN PHỐI (không bao giờ một lần bốc)
python experiments/feasibility/control_chain_distribution.py --draws 200

# --- hình cho bài báo
python experiments/model_eval/make_figures.py                  # -> docs/figures/*.pdf

# --- probe (CẦN hệ đang chạy): đo k, method, và D_feat khi hệ rảnh
python experiments/collect/probe_feature_chain.py --features browse,cartsum,quickadd,express \
       --n 20 --n-latency 200 --save data/processed/frozen/k_measured_v2.json

pytest tests/ -q                                    # 44 test
```

📂 Mục lục toàn bộ 82 script: [`experiments/README.md`](../experiments/README.md)

---

## 11. Việc còn lại, xếp theo mức đe doạ việc được nhận

| # | Việc | Vì sao |
|---|---|---|
| 1 | **Làm lại con số signal gate có nguồn gốc** | nó nằm trong **Abstract** và hiện không tái lập được |
| 2 | **Một vòng đo tiến cứu mới** | trụ cột hiện là **n=2**; không sửa được bằng phân tích |
| 3 | **Phát hành dữ liệu** (Zenodo + DOI) | testbed là đóng góp chính; `data/raw/` đang gitignore |
| 4 | Sửa `§rq6-latency` | đang viết "không có hướng khắc phục" — **không còn đúng** |
| 5 | Đối chứng chain‑sai mạnh hơn (loại gateway) | đối chứng hiện tại không đủ sức bác bỏ |
| 6 | Báo cáo sai số điểm gãy theo **req/s tuyệt đối** | sai số tương đối không so sánh được giữa các ô |
