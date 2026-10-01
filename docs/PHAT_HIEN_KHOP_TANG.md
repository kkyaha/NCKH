# Khớp tầng: phát hiện, cơ chế, và cải tiến

> Tài liệu độc lập. Mọi con số đều truy được về một script và một CSV trong repo.
> Tái lập toàn bộ: bốn lệnh ở mục 8.

---

## 1. Phát hiện, một câu

> **Độ chính xác định vị nguyên nhân bị quyết định bởi TẦNG metric được soi, không bởi thuật toán xếp hạng — và tầng đúng là tầng mà nguyên nhân nằm.**

Mô hình nhân quả của một hệ microservice có **ba** tầng quan sát, không phải hai:

| tầng | cột | nghĩa |
|---|---|---|
| **nhu cầu** | `<svc>_workload` | số request mỗi giây tới service |
| **tài nguyên** | `<svc>_cpu` | CPU service tiêu thụ |
| **độ trễ** | `<svc>_latency-50` | thời gian phục vụ |

Và ba loại cạnh, mỗi loại nằm **trong một tầng** (`scm_graph_builder.py:70-79`):

| cạnh | dạng | nghĩa |
|---|---|---|
| lan truyền nhu cầu | `workload → workload` | request đi theo chuỗi gọi |
| chuyển hoá cục bộ | `workload → cpu` / `mem` / `latency` | mỗi request tốn tài nguyên |
| **tranh chấp** | **`cpu → cpu`** | service A nghẽn làm chậm B dù request của B không đổi |
| **lan truyền độ trễ** | **`latency → latency`** (ngược) | callee chậm làm caller chậm |

Mỗi loại nguyên nhân sinh ra tín hiệu **trên tầng của nó**, rồi lan **trong** tầng đó. Nên:

> Một phương pháp định vị chỉ đúng khi **tầng nó soi** trùng **tầng mà nguyên nhân nằm**.

## 1b. Hình thức hoá

**Định nghĩa (đáp ứng tầng).** Với nguyên nhân tại service `x` và tầng `L`:

```latex
r_L(x) \;=\; \frac{\bigl|\,\mathbb{E}[L_x \mid \text{nguyên nhân}] - \mathbb{E}[L_x]\,\bigr|}{\sigma(L_x)}
```

**Mệnh đề 1 (dạng độ lớn).** Tầng `L` mang thông tin định vị `x` khi và chỉ khi `r_L(x)` vượt khỏi độ trải của `r_L(v)` trên các `v ≠ x`. Trong một SCM tuyến tính, `r_L(x)` là tổng tích hệ số theo mọi đường có hướng từ nguyên nhân tới `L_x`; một tầng **không có đường nào** tới nguyên nhân có `r_L(x) = 0` và **đúng mức ngẫu nhiên**.

> Chứng minh: với can thiệp cứng, Pearl (2009) §3.4 Theorem 3.4.1 Rule 3 cho `P(L_x | do(·)) = P(L_x)` khi `L_x` d-separated khỏi điểm can thiệp trong đồ thị đã cắt. Với can thiệp mềm (tiêm lỗi — đổi *cơ chế*, không đặt cứng giá trị), tính bất biến do **modularity** cho cùng kết luận: đổi cơ chế của một node chỉ ảnh hưởng node đó và hậu duệ của nó (Pearl §1.4.2; Peters–Bühlmann–Meinshausen, JRSS‑B 2016).

**Hệ quả (dạng tỉ số, kiểm được).** Định nghĩa **tỉ số rò rỉ** giữa tầng lệch `D` và tầng đúng `R`:

```latex
\rho_{\text{rò rỉ}} \;=\; r_D(x) \,/\, r_R(x)
```

Đo được tại service bị tiêm, trung vị:

| hệ | nhóm lỗi | `r_workload` | `r_cpu` | `r_latency` | **ρ rò rỉ** (D/R) |
|---|---|---|---|---|---|
| SockShop | tài nguyên | 0,086 | **184,8** | 12,2 | **5×10⁻⁴** |
| TrainTicket | tài nguyên | 0,187 | **31,0** | 22,8 | **6×10⁻³** |
| SockShop | mạng | 1,125 | 0,826 | **1035,6** | 1,36 |
| TrainTicket | mạng | 0,298 | 0,468 | **9,99** | 0,64 |

**Điều kiện phạm vi, phát biểu đúng.** Không phải "không có cạnh `R → D`" (một giả định nhị phân mà không hệ thật nào thoả) mà là **một bất đẳng thức về độ lớn**:

```latex
\rho_{\text{rò rỉ}} \ll 1
```

Với lỗi tài nguyên, `ρ` là **5×10⁻⁴** và **6×10⁻³** — ba bậc độ lớn. Đó là vì sao tầng nhu cầu đúng mức ngẫu nhiên dù cạnh `R → D` **có tồn tại** (xem mục 7).

> **Lưu ý về tính không vòng tròn.** Bảng mẫu cạnh **không có** mẫu `cpu → workload`, nên đồ thị **không thể** chứa cạnh đó — giả định của Mệnh đề 1 là **do xây dựng**, không do phát hiện. Vì vậy thứ tự lập luận trong bài phải là: **(i)** tiền đề thực nghiệm đo từ **dữ liệu thô**, độc lập hoàn toàn với đồ thị (mục 7); **(ii)** lựa chọn mô hình được (i) biện minh; **(iii)** lý thuyết áp vào mô hình đó; **(iv)** xác nhận hệ quả. Trình bày theo thứ tự lý thuyết‑trước sẽ che mất bước (i) và bị bắt ngay.

## 2. Bằng chứng chính: 90 lần tiêm lỗi thật, không vòng tròn

Dữ liệu: RCAEval **RE2-SS**, 5 service × 6 loại lỗi × 3 lần lặp = **90 run**. Service bị tiêm
lấy từ **tên thư mục** (`catalogue_cpu`, `carts_mem`, …) — **không suy từ đồ thị nào**.
Đáp án do can thiệp tạo ra, không phải suy từ tương quan.

### 2.1 Định vị theo tầng, tổng hợp

| tầng được soi | top‑1 | top‑3 | hạng trung bình |
|---|---|---|---|
| **tài nguyên (`_cpu`)** | **76,7%** | **86,7%** | **1,7** |
| nhu cầu (`_workload`) | 13,3% | 54,4% | 3,7 |
| *ngẫu nhiên (7 ứng viên)* | *14,3%* | *42,9%* | *4,0* |

Tầng nhu cầu **đúng bằng ngẫu nhiên**.

### 2.2 Theo loại lỗi — đây là chỗ cơ chế lộ ra

| loại lỗi | nguyên nhân nằm ở tầng | soi `_workload` | soi `_cpu` | soi `_latency` |
|---|---|---|---|---|
| cpu | **tài nguyên** | 20,0% | **100,0%** | 60,0% |
| mem | **tài nguyên** | 6,7% | **100,0%** | 86,7% |
| socket | **tài nguyên** | 6,7% | **100,0%** | 80,0% |
| disk | **tài nguyên** | 20,0% | **93,3%** | 53,3% |
| **loss** | **độ trễ** | 26,7% | 33,3% | **86,7%** |
| **delay** | **độ trễ** | 0,0% | 13,3% | **86,7%** |

| hệ | nhóm | `_workload` | `_cpu` | `_latency` |
|---|---|---|---|---|
| SockShop | **tài nguyên** (n=60) | 13,3% | **98,3%** | 70,0% |
| SockShop | **mạng** (n=30) | 13,3% | 43,3% | **86,7%** |
| TrainTicket | **tài nguyên** (n=60) | 5,0% | **96,7%** | 46,7% |
| TrainTicket | **mạng** (n=30) | 3,3% | 3,3% | **33,3%** (top‑3: **70,0%**) |

Nguồn: `experiments/legacy/rq6_layer_localization_real_faults.py` → `rq6_layer_localization_real_faults.csv`

### 2.3 Chiều ngược lại: can thiệp nhu cầu

Khi nguyên nhân là **can thiệp trên tầng nhu cầu** (`do(gateway_workload = 2,5×)`), tầng
thắng **đảo lại**:

| phép chấm | | tầng **nhu cầu** | tầng tài nguyên |
|---|---|---|---|
| theo đồ thị gọi *(vòng tròn cấu trúc)* | TrainTicket | **96,5%** · AUC **0,989** | 68,8% · AUC 0,630 |
| theo dữ liệu quan sát *(không đồ thị)* | TrainTicket | **68,0%** · AUC **0,851** | 63,0% · AUC 0,725 |
| | SockShop | **100,0%** · AUC **1,000** | 66,7% · AUC 0,874 |
| **Spearman(dịch chuyển, tương quan quan sát)** *(dữ liệu đối dữ liệu, không ngưỡng, không đồ thị)* | TrainTicket | **+0,790** | +0,471 |
| | SockShop | **+0,771** | +0,234 |

Phép thứ ba là phép **sạch nhất**: hai phía đều từ dữ liệu. Ưu thế của tầng nhu cầu thu
hẹp khi khử vòng tròn (96,5% → 68,0%) nhưng **không mất**, và giữ được trên **cả hai hệ**.

Nguồn: `experiments/legacy/rq6_localization_decircularized.py` → `rq6_localization_decircularized.csv`

### 2.4 Hai chiều gộp lại

| nguyên nhân nằm ở tầng | tầng định vị được | tầng thất bại |
|---|---|---|
| tài nguyên (lỗi cpu/mem/disk/socket) | **tài nguyên 98,3%** | nhu cầu 13,3% = ngẫu nhiên |
| nhu cầu (can thiệp workload) | **nhu cầu ρ +0,79** | tài nguyên ρ +0,47 |
| **độ trễ (lỗi delay/loss)** | **độ trễ — 86,7%** | workload 0–26,7% · CPU 13,3–33,3% |

Ba dòng này là lý thuyết khép kín: **ba loại nguyên nhân, ba tầng, mỗi loại định vị được trên
đúng tầng của nó.** Không còn trường hợp ngoại lệ nào phải thú nhận.

> **Sửa một kết luận sai của bản trước.** Bản đầu của tài liệu này ghi rằng lỗi `delay` *"nằm
> trên một tầng mà mô hình không biểu diễn"*. **Sai.** Mẫu cạnh thứ sáu là `latency-50 →
> latency-50`, nên mô hình **có** tầng độ trễ với lan truyền riêng. `delay` không định vị được
> chỉ vì **chưa ai thử tầng đó** — thử rồi thì nó đi từ 13,3% (ngẫu nhiên) lên **86,7%**.

---

## 2b. Đơn vị quy gán: lỗi mạng nằm ở **cạnh**, không ở **node**

Lỗi tài nguyên định vị được trên tầng CPU với **đơn vị node** (98,3%). Lỗi mạng thì không —
trên Train Ticket chỉ **33,3%** dù đã soi đúng tầng latency. Chẩn đoán:

| node dẫn đầu là ai | SockShop | TrainTicket |
|---|---|---|
| **chính service bị tiêm** | 13/15 | 10/30 |
| **tổ tiên** của nó | — | **8/30** |
| **caller trực tiếp** | — | **6/30** |
| không liên quan | 2/15 | 6/30 |

Và độ lớn giải thích vì sao:

| | hạng của service bị tiêm | dịch chuyển: bị tiêm | node dẫn đầu | tỉ lệ |
|---|---|---|---|---|
| SockShop | **1/7** | 1035,6 σ | 1397,2 σ | **1×** |
| TrainTicket | 2/28 | 10,0 σ | 316,7 σ | **32×** |

**Mất gói và độ trễ xảy ra trên CẠNH mạng**, nên tín hiệu độ trễ lớn nhất ở **node quan sát
cạnh đó** (caller / tổ tiên), không ở service bị tiêm — service đó **không hỏng**, **đường tới
nó** hỏng. Sock Shop nông (một gateway) nên node bị tiêm vẫn áp đảo; Train Ticket có chuỗi gọi
sâu nên tín hiệu dồn lên thượng nguồn.

### Chấm theo đúng đơn vị

| cách chấm | SockShop t1 | TrainTicket t1 | TrainTicket t3 |
|---|---|---|---|
| **node** (service bị tiêm là top‑1) | 86,7% | **33,3%** | 70,0% |
| **cạnh** (top‑1 → service bị tiêm là một cạnh) | 96,7% | 53,3% | 66,7% |
| **đường** (service bị tiêm là hậu duệ của top‑1) | **96,7%** | **80,0%** | **86,7%** |

Theo từng loại lỗi trên Train Ticket: `delay` **60,0% → 93,3%**; `loss` **6,7% → 66,7%**.

> **Hệ quả về giao thức đánh giá.** Nhãn của RCAEval gán lỗi mạng theo **service**, nhưng thực
> thể bị hỏng là **cạnh** tới service đó. Chấm theo đơn vị node là **chấm sai đơn vị** — và sai
> số đó **lớn lên theo độ sâu chuỗi gọi** (1× ở Sock Shop → 32× ở Train Ticket). Điều này áp
> dụng cho mọi ai dùng bộ dữ liệu đó, không riêng bài này.

### Lý thuyết khép kín: ba vị trí nguyên nhân, ba vị trí tín hiệu

| nguyên nhân nằm ở | tín hiệu lớn nhất ở | tầng | đơn vị | độ chính xác |
|---|---|---|---|---|
| **node**, tầng tài nguyên | chính node đó | CPU | node | **98,3%** |
| **node**, tầng nhu cầu | hậu duệ của node đó | workload | node | ρ **+0,79** |
| **cạnh** giữa hai node | **node quan sát cạnh** (thượng nguồn) | latency | **đường** | **80,0% / 86,7%** |

> **Soi đúng tầng, và quy gán đúng đơn vị.** Tầng quyết định *ở đâu* có tín hiệu; đơn vị quy gán
> quyết định *đọc tín hiệu đó thế nào*.

Nguồn: `experiments/legacy/rq6_attribution_unit.py` → `rq6_attribution_unit.csv`

## 3. Cơ chế — vì sao tầng tài nguyên bị nhiễu

Soi trực tiếp hệ số NNLS đã fit, **không qua Shapley**, trên bốn node Train Ticket:

| node đích | cha nhu cầu (thang hiệu dụng) | cha tranh chấp | tỉ lệ | tỉ phần tranh chấp |
|---|---|---|---|---|
| `ts-order-service_cpu` | 0,033 | 1,533 | **46,3×** | **97,9%** |
| `ts-seat-service_cpu` | 0,031 | 2,040 | **66,8×** | **98,5%** |
| `ts-travel-service_cpu` | 0,263 | 2,646 | 10,1× | 91,0% |
| `ts-user-service_cpu` | 0,028 | 0,233 | 8,3× | 89,2% |

Trung vị: cha tranh chấp gấp **28,2×** cha nhu cầu, chiếm **94,4%** thang hiệu dụng.

CPU của một service bị chi phối bởi **CPU của các service khác**, không bởi **lượng request
của nó**. Đó chính là nhiễu.

Nguồn: `rq6_hop1_node_diagnostic.csv`

---

## 4. Cải tiến — và đây là điểm mạnh nhất

Câu hỏi: 98,3% là **do thống kê tốt hơn** hay **do soi đúng tầng**? Tách bằng lưới 2×2×2
trên **cùng 90 run, cùng đáp án, cùng cửa sổ trước/sau**:

- **thống kê**: BARO (`RobustScaler` median/IQR, `max|z|`, dùng nguyên văn luật của RCAEval `e2e/baro.py`) vs MEANSHIFT (`|Δtrung bình| / std trước`)
- **tầng**: chỉ cột `_cpu` vs mọi cột metric
- **ứng viên**: 7 service ứng dụng vs mọi service trong cột (~15)

### 4.1 Lỗi tài nguyên, top‑1

| ứng viên | tầng | thống kê | top‑1 | top‑3 | ngẫu nhiên |
|---|---|---|---|---|---|
| **7 app** | **chỉ `_cpu`** | **BARO** | **98,3%** | **100,0%** | 14,3% |
| **7 app** | **chỉ `_cpu`** | **MEANSHIFT** | **98,3%** | **100,0%** | 14,3% |
| 7 app | mọi cột | MEANSHIFT | 91,7% | 100,0% | 14,3% |
| 7 app | mọi cột | BARO | 81,7% | 98,3% | 14,3% |
| ~15 svc | chỉ `_cpu` | MEANSHIFT | 98,3% | 98,3% | 6,7% |
| ~15 svc | chỉ `_cpu` | BARO | 95,0% | 98,3% | 6,7% |
| ~15 svc | mọi cột | MEANSHIFT | 91,7% | 96,7% | 6,7% |
| **~15 svc** | **mọi cột** | **BARO** | **11,7%** | 85,0% | 6,7% |

### 4.2 Tách yếu tố

| | |
|---|---|
| **tầng đóng góp** (chỉ `_cpu` − mọi cột) | BARO **+16,7 đ** · MEANSHIFT **+6,7 đ** |
| **thống kê đóng góp** (MEANSHIFT − BARO) | trên tầng **đúng**: **+0,0 đ** · trên tầng **sai**: +10,0 đ |

**Trên tầng đúng, hai thống kê bằng nhau tuyệt đối (98,3% = 98,3%).** Thống kê chỉ quan
trọng khi tầng đã sai.

### 4.3 Phát biểu của cải tiến

> **Không phải "phương pháp mới tốt hơn BARO". Mà là: BARO — một baseline đã xuất bản —
> đạt 98,3% top‑1 / 100% top‑3 khi chỉ soi tầng tài nguyên, thay vì 11,7% khi trộn mọi
> tầng metric. Chênh 86,6 điểm, hoàn toàn do một lựa chọn cấu hình.**

Tách được nguồn của 86,6 điểm đó:

| bước | top‑1 |
|---|---|
| 7 app · chỉ `_cpu` | **98,3%** |
| → mở rộng lên ~15 ứng viên | 95,0% *(−3,3 đ)* |
| → thêm mọi cột metric | **11,7%** *(−83,3 đ)* |

Thủ phạm **không** phải số ứng viên mà là **trộn các tầng metric**. Khi xếp theo `max|z|`
trên mọi cột, một cột `_mem` hoặc `_latency` nhiễu ở một service bất kỳ dễ vượt cột `_cpu`
của service thật sự bị tiêm. **Trộn tầng thì tầng nhiễu nhất thắng.**

Nguồn: `experiments/legacy/rq6_baro_matched_comparison.py` → `rq6_baro_matched_comparison.csv` (720 dòng = 90 run × 8 điều kiện)

---

## 5. Hệ quả: chẩn đoán lại thất bại của quy gán nhân quả

Trong thiết lập gốc của dự án, quy gán **can thiệp trên tầng nhu cầu** rồi **quy gán trên
tầng tài nguyên** — **lệch tầng**. Hệ quả đo được:

| khoảng cách | strict (chỉ cạnh nhu cầu) | kèm cạnh tranh chấp |
|---|---|---|
| 1 hop (n=40) | 25,0% | 40,0% |
| 2 hop (n=40) | 2,5% | **92,5%** |
| 3 hop (n=10) | **0,0%** | **90,0%** |

Spearman(hop, strict) = **−1,0**. Không phải "tín hiệu loãng theo khoảng cách" — mà công
**đi sang cạnh tranh chấp**. Trong 55 ca được phục hồi, cạnh tranh chấp mang **59,7%**
khối lượng quy gán so với **1,1%** của cạnh nhu cầu (**15×**). Ba chế độ tách sạch:

| chế độ | n | tỉ phần tranh chấp |
|---|---|---|
| nhu cầu chi phối → strict đúng | 11 | 21,9% |
| **tranh chấp chi phối → chỉ upstream đúng** | **55** | **59,7%** |
| cục bộ chi phối → cả hai sai | 24 | 36,5% |

Và **không phải khuyết điểm của một phương pháp**: BARO (không nhân quả) cũng đạt 40–50%
ở đúng node Shapley tệ nhất; cả hai 100% ở ca cục bộ.

Nguồn: `rq6_hop_distance_diagnostic.csv` (lọc `scenario == 'tier1_driven'`), `rq6_baro_comparison.csv`

---

## 6. Quy mô khuếch đại chi phí lệch tầng

| | Sock Shop (7 svc, 28 node) | Train Ticket (28 svc, 112 node) |
|---|---|---|
| tỉ phần **nhu cầu** | **81,0%** | **15,7%** |
| tỉ phần **tranh chấp** | 0,3% | **33,9%** |
| tỉ phần cục bộ | 18,7% | 46,4% |
| độ chính xác quy gán | **90,0%** | **0,0%** |

Hệ càng lớn, tranh chấp càng nhiều, tầng tài nguyên càng nhiễu → lệch tầng càng đắt. Đây
cũng là lý do phương pháp xác nhận trên benchmark nhỏ không chuyển được sang hệ lớn.

**Và tỉ phần đó dự đoán được độ chính xác:** Pearson **0,957** (p = 1,5×10⁻⁵, n = 10, hai
hệ). Ngưỡng 50% tách hoàn hảo — trên ngưỡng độ chính xác trung vị **100%**, dưới **0%**.
Tính được từ cơ chế đã fit, **không cần đáp án**, nên dùng được lúc triển khai:

> Trước khi quy gán, đo tỉ phần nhu cầu. Dưới ngưỡng thì **từ chối trả lời**.

Nguồn: `rq6_tier_decomposition_v2_summary.csv`

---

## 7. Điều kiện phạm vi — đo được, không chỉ khai báo

Mệnh đề 1 cần `ρ rò rỉ ≪ 1`. Kiểm trực tiếp bằng **dấu và độ lớn** của dịch chuyển workload tại
service bị tiêm — phép đo này dùng **dữ liệu thô** trước/sau, **không dùng đồ thị**, nên nó là
tiền đề độc lập chứ không phải hệ quả của mô hình.

| hệ | nhóm lỗi | n | dịch chuyển trung vị | tăng/giảm | Wilcoxon `p` | `D` đóng? |
|---|---|---|---|---|---|---|
| SockShop | **tài nguyên** | 60 | **+0,017 σ** | 32/28 | **0,230** | ✅ đóng |
| SockShop | mạng | 30 | −1,125 σ | 2/28 | 9×10⁻⁹ | ❌ rò rỉ |
| **TrainTicket** | **tài nguyên** | 60 | **−0,107 σ** | 22/38 | **0,0018** | ❌ **rò rỉ** |
| TrainTicket | mạng | 30 | −0,298 σ | 7/23 | 4×10⁻⁵ | ❌ rò rỉ |

**Train Ticket có cạnh `R → D` thật**, kể cả với lỗi tài nguyên — dấu **âm** (service chậm thì
hoàn thành ít request hơn), tức cơ chế **chặn**, không phải **retry**.

Nhưng kết luận **không đổ**: dù cạnh đó tồn tại, tầng nhu cầu **vẫn đúng mức ngẫu nhiên**
(5,0% vs 3,6%, `p = 0,475`). Rò rỉ **có thật nhưng quá yếu** — `ρ = 6×10⁻³`, ba bậc độ lớn.

> Đó là lý do Mệnh đề 1 phải phát biểu dạng **bất đẳng thức độ lớn**, không phải dạng vắng mặt
> cạnh: điều kiện vắng mặt **không hệ thật nào thoả**, còn điều kiện độ lớn thì thoả rộng rãi.

**Phép kiểm ai cũng chạy được trước khi tin kết luận:** tiêm một lỗi tài nguyên, đo dấu dịch
chuyển workload. Khác 0 đáng kể thì tầng nhu cầu không đóng — nhưng vẫn dùng được nếu tỉ số rò
rỉ nhỏ.

## 8. Tiên đoán điểm — kiểm nhị thức

Lý thuyết nói tầng lệch đạt **đúng** mức ngẫu nhiên, không phải "kém hơn". So hai con số bằng mắt
là không đủ; đây là `p` cho `H₀: tỉ lệ = 1/k`:

| hệ | nhóm lỗi | tầng lệch | tỉ lệ | ngẫu nhiên | `p` | kết luận |
|---|---|---|---|---|---|---|
| SockShop | tài nguyên | workload | 13,3% | 14,3% | **1,000** | **= ngẫu nhiên** |
| SockShop | mạng | workload | 13,3% | 14,3% | **1,000** | **= ngẫu nhiên** |
| TrainTicket | tài nguyên | workload | 5,0% | 3,6% | **0,475** | **= ngẫu nhiên** |
| TrainTicket | mạng | workload | 3,3% | 3,6% | **1,000** | **= ngẫu nhiên** |

Đối chiếu, tầng **đúng**: `p = 2×10⁻⁶⁸` (SockShop, CPU) và `9×10⁻¹⁰⁴` (TrainTicket, CPU).

Bốn lần, hai hệ, không bác được. Đây là tiên đoán mạnh nhất của lý thuyết vì nó là **một điểm**,
không phải một chiều.

## 9. Giới hạn — viết trước, đừng chờ reviewer

1. **Mục 2 là định vị lỗi ĐÃ xảy ra**, không phải dự phóng. Nó khử vòng tròn cho câu hỏi
   **tầng**, không chứng minh gì về dự đoán tính năng chưa tồn tại.
2. **Trục quy mô chỉ có hai điểm** (7 và 28 service). Mục 6 là quan sát nhất quán với một
   hiệu ứng quy mô, **chưa phải quy luật**. `data/raw/RE2-OB` (~11 service, 90 kịch bản) là
   điểm thứ ba, nhưng `CapacityAgent` chưa có nhánh Online Boutique.
3. **Tiêu chí khả nhận: n = 10 node**, ngưỡng 50% fit trong mẫu → là **mô tả**, chưa phải
   ngưỡng vận hành. Cần kiểm trên tập giữ lại.
4. **Mục 2.3 phép (A) có vòng tròn cấu trúc** — cạnh `workload→workload` dựng từ đồ thị
   gọi, mà đáp án cũng là đồ thị gọi. Đã khử bằng phép (B) và (C); ưu thế thu hẹp 96,5% →
   68,0% và phải báo cả ba.
5. **Cạnh tranh chấp là cạnh học** — nhưng đã kiểm an toàn OOD **trước** khi có các kết quả
   này (`backpressure_edge_ood_safety.csv`), và độ lớn xác nhận độc lập bằng NNLS.
6. **Đồ thị gọi khai quá rộng**: Train Ticket có 17 service reachable theo đồ thị nhưng chỉ
   **10** theo dữ liệu quan sát, trùng nhau **9**. Một phần thất bại quy gán do **đồ thị đầu
   vào**, không do phương pháp.
7. **Lỗi mạng cần đơn vị quy gán là CẠNH, không phải node** (mục 2b). Chấm theo node cho
   33,3% trên Train Ticket; chấm theo đường cho **80,0%**. Nhưng `loss` vẫn là loại khó nhất
   (66,7% theo đường, 73,3% top‑3) — chưa đạt mức của lỗi tài nguyên (98,3%), và vì sao thì
   chưa giải thích hết.

---

## 10. Tái lập

```bash
# (1) khớp tầng bằng nhãn tiêm lỗi thật — 90 run RE2-SS
python experiments/legacy/rq6_layer_localization_real_faults.py

# (2) lưới khớp với BARO: tầng vs thống kê vs ứng viên
python experiments/legacy/rq6_baro_matched_comparison.py

# (3) chiều can thiệp nhu cầu, ba phép chấm, hai hệ
python experiments/legacy/rq6_localization_decircularized.py --repeats=10

# (4) so bốn tín hiệu xếp hạng trên 27 service Train Ticket
python experiments/legacy/rq6_rank_signal_comparison.py --repeats=10

# (5) BA TẦNG + kiểm nhị thức + điều kiện phạm vi, trên CẢ HAI hệ
python experiments/legacy/rq6_three_layer_and_scope.py

# (6) ĐƠN VỊ QUY GÁN: node vs cạnh vs đường, cho lỗi mạng, hai hệ
python experiments/legacy/rq6_attribution_unit.py
```

Phụ trợ, cần cho mục 5–6 và cho giao thức đánh giá:

```bash
python experiments/legacy/recalibrate_risk_threshold.py          # hiệu chỉnh ngưỡng rủi ro
python experiments/legacy/rq6_topology_check.py --calibrated     # có đối chứng dương
python experiments/legacy/rq6_attribution_validity.py --calibrated
```

> **Ghi chú về ngưỡng**: ngưỡng gán cứng `z ≥ 1,0` trong `future_rca.py` **không đạt được**
> (z max quan sát 0,448), khiến `flagged_rate = 0` ở **mọi** nhóm kể cả node bị tiêm — nên
> tuyên bố "zero false positive" trước đây là **rỗng**, không có đối chứng dương. Hiệu chỉnh
> từ nửa null giữ lại (`z_warn=0,20`, `z_crit=0,40`) cho **0% FP** trên nửa held-out **và**
> **100% phát hiện** node bị tiêm. Mặc định vẫn là 1,0/2,0 để mọi số đã báo tái lập bit‑for‑bit.

---

## 11. Việc còn lại

Bốn việc trong các bản trước **đã làm xong**: định vị trên tầng độ trễ (mục 2), kiểm nhị thức cho
tiên đoán điểm (mục 8), điều kiện phạm vi trên hai hệ (mục 7), và **đơn vị quy gán cho lỗi mạng**
(mục 2b). Còn lại:

| | việc | củng cố | chi phí |
|---|---|---|---|
| 1 | **hệ thứ ba** — nối Online Boutique vào `CapacityAgent` (`RE2-OB` đã có 90 kịch bản, ~11 service) | mục 6 từ 2 → 3 điểm trên trục quy mô | trung |
| 2 | ~~giải thích `loss` trên Train Ticket~~ — **đã xong** (mục 2b): nguyên nhân là **sai đơn vị quy gán**, 6,7% → 66,7% khi chấm theo đường | — | — |
| 3 | **kiểm tiên đoán "tập cha"** — dương tính giả của tầng CPU phải **chính là** node có cạnh tranh chấp từ node hậu duệ | tiên đoán **cấu trúc** sắc nhất, chưa kiểm | trung |
| 4 | **kiểm ngưỡng 50% trên tập giữ lại** | tiêu chí khả nhận thành ngưỡng vận hành | thấp |
| 5 | **mở rộng phép soi NNLS từ 4 → ≥10 node** cả hai hệ | mục 3 hết vòng tròn qua Shapley | thấp |

Việc **2** đáng làm trước: nó là ngoại lệ duy nhất còn lại, và nếu giải thích được thì lý thuyết
không còn trường hợp nào phải thú nhận. Việc **1** vẫn là điểm yếu nặng nhất về tính tổng quát.
