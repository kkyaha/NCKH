# Bộ khung paper V2 — bản để duyệt

> ⚠️ **TÀI LIỆU THEO HỆ ĐÁNH SỐ RQ CŨ (RQ1–RQ12).** Số RQ trong file này **không khớp**
> khung 5 RQ hiện tại của `docs/paper_draft.tex`. Xem bảng dịch ở đầu `README.md` gốc.
> Giữ lại làm hồ sơ gốc; **không dùng làm nguồn số liệu**.

> Dựng lại sau toàn bộ kết quả phiên 2026-09-13/14. Mỗi tuyên bố dưới đây đều
> trỏ tới bằng chứng cụ thể. Ô ⚠️ là chỗ còn thiếu.

---

## Tiêu đề đề xuất

> **Pre-Mortem Capacity Forecasting for Unbuilt Software Requirements:
> A Verified LLM Front-End and What Causal Structure Does (and Does Not) Buy**

Giữ khung bài toán — thứ thật sự mới — nhưng nói thẳng nửa sau là khảo sát,
không phải tuyên bố thắng lợi.

*Phương án ngắn hơn nếu venue giới hạn độ dài:*
**"Verified Requirement-to-Intervention Translation for Microservice Capacity
Forecasting"**

---

## Luận đề một câu

Dịch một requirement ngôn ngữ tự nhiên thành một can thiệp capacity **có bảo
đảm cấu trúc chứng minh được**; cho thấy mô hình capacity dự báo được **khi và
chỉ khi** dữ liệu đánh giá có tín hiệu tải; và đặc tả chính xác **cấu trúc nhân
quả mua được gì** — câu trả lời là: cấu trúc đồ thị có giá trị, toán tử $do$
thì không, trong cấu hình của bài toán này.

---

## Ba đóng góp, xếp theo sức mạnh bằng chứng

1. **Bộ xác minh có bảo đảm chứng minh được** cho đầu ra LLM (Proposition 1),
   kèm bằng chứng adversarial rằng nó *gánh việc* chứ không rỗng, và một lớp
   abstention hiệu chỉnh conformal với cận khả thi được nêu trung thực.

2. **Một giao thức đánh giá có chế độ thất bại** — signal gate + baseline hằng
   số + negative control — và phát hiện rằng benchmark RCA hiện hành **không
   dùng được** cho capacity forecasting.

3. **Đặc tả ranh giới của cấu trúc nhân quả**: đồ thị Tier-1 có giá trị đo
   được; toán tử $do$ quy về điều kiện hoá trong cấu hình này, và chúng tôi chỉ
   ra điều kiện chính xác để nó không quy về.

---

## Cấu trúc section

| § | Nội dung | Trạng thái |
|---|---|---|
| I | Introduction — khoảng trống, đóng góp | viết lại |
| II | Background & Related Work + **bảng định vị** + **tiểu mục "khi nào $do$ có nội dung"** | mở rộng |
| III | Approach: pipeline, **verifier (Prop 1, Alg 1)**, abstention, mô hình capacity hai tầng | sắp xếp lại |
| IV | Experimental Setup — **ba dataset, giải thích vì sao tách** | mới |
| V | **RQ1–RQ3** — ràng buộc LLM (Sock Shop, Train Ticket) | phần lõi |
| VI | **RQ4** — mô hình capacity (Alibaba) | **viết mới** |
| VII | **RQ5** — cấu trúc nhân quả mua gì + tiểu mục hệ quả vận hành | viết mới |
| VIII | Discussion — hàm ý cho văn liệu causal-AIOps và cho benchmark | mới |
| IX | Threats (Wohlin) | có sẵn, cập nhật |
| X | Conclusion | viết lại |

---

## Vì sao ba dataset — phải giải thích tường minh ở §IV

**Không bộ dữ liệu nào có cả hai thứ cần:**

| | Tên service có nghĩa | Biên độ tải rộng |
|---|---|---|
| Dataset RCA (RCAEval, Nezha…) | ✅ | ❌ baseline ổn định *cố ý* |
| Trace production (Alibaba, Google) | ❌ hash | ✅ |
| Benchmark suite (DeathStarBench…) | ✅ | ✅ *nhưng phải tự chạy* |

Xung đột cấu trúc: tính chất làm benchmark tốt cho RCA (baseline ổn định) chính
là tính chất làm nó vô dụng cho forecasting. Đây **tự nó là một phát hiện**, và
là lý do thứ hai (độc lập với thiếu ground truth) giải thích vì sao văn liệu ở
lại phía hồi cứu.

| Nửa | Dataset | Lý do |
|---|---|---|
| Parser + guardrail | Sock Shop, Train Ticket | cần tên service có nghĩa |
| Mô hình capacity | **Alibaba v2021** | cần biên độ tải |
| Kiểm định can thiệp | RCAEval fault injection | đúng mục đích của nó |

---

## BỘ 5 RQ (bản đã gộp — 11 RQ ban đầu là quá nhiều)

> Chuẩn TSE/TOSEM là 3–5 RQ. Bản dưới gộp 11 → 5 mà **không mất bằng chứng
> nào**; mỗi RQ có nhiều phát hiện con thay vì mỗi phát hiện một RQ.

---

### RQ1. Đầu ra LLM có ràng buộc được bằng bảo đảm vừa *chứng minh được* vừa *gánh việc*?

*(gộp RQ1 + RQ2 cũ)* — **§V**, dataset: Sock Shop

| Bằng chứng | Số |
|---|---|
| Proposition 1 + Corollary 1 | chứng minh, độc lập backend |
| PBVR / SHR | $0{,}0\%$, Wilson CI $[0{,}0;2{,}5]\%$, $n{=}150$ |
| Gateway fallback, bộ adversarial | **18/18** |
| Inner clamp, 6 chiến lược thao túng | **33/36 (91,7%)** |
| Tỉ lệ kích hoạt trên bộ lành tính | $0\%$ |
| ⚠️ Đa backend | **CHƯA CÓ** |

**KHÔNG** tuyên bố GMR $0\%$ là tác dụng của guard (confounded với prompt).

Hai nửa của cùng một câu hỏi: bảo đảm *tồn tại* (chứng minh) và bảo đảm *có
việc để làm* (adversarial). Tách ra thì nửa sau trông như phụ lục.

---

### RQ2. Abstention tốn gì, và bảo đảm *khả thi* đến đâu ở cỡ mẫu này?

*(RQ3 cũ)* — **§V**, dataset: Sock Shop

| Bằng chứng | Số |
|---|---|
| Out-of-scope lọt auto-pass | $0/21$ |
| Vào vùng review | $16/81$ ($19{,}8\%$) |
| Từ chối oan | $3/60$ ($5{,}0\%$), giảm từ $16{,}7\%$ |
| **Cận α khả thi** | $1/(n{+}1) = 4{,}5\%$ — **không phải $0{,}01$** |
| Bộ adversarial đánh bại tiêu chí keyword | **$30/30$** |

Bảo đảm bị chặn bởi **cỡ mẫu**, không bởi α đã chọn. Công bố cả thất bại.

---

### RQ3. Chuyển sang hệ thứ hai được không, và việc chuyển *phơi ra* gì?

*(RQ4 cũ)* — **§V**, dataset: Train Ticket

| Bằng chứng | |
|---|---|
| Taxonomy sinh từ đồ thị | 10 archetype, Jaccard TB 0,21, không cặp trùng |
| Topology đa gateway | **14 gateway** (Sock Shop: 1) |

**Ba lỗi mà kiểm chứng một-hệ không thể phơi ra:**

1. Blast radius sinh sai — BFS từ gateway out-degree 15 → mọi archetype phủ
   28/68 node, gần trùng nhau
2. Guard gateway gửi **mọi** yêu cầu khách hàng tới `ts-admin-order-service`
3. **`_guard_core_services` hardcode `['front-end']`** → tự chèn service không
   tồn tại → **vi phạm chính Proposition 1**

Lỗi #3 là hiểu biết thật: *một verifier được chứng minh đúng vẫn cần kiểm chứng
cài đặt.* Nêu nó làm RQ1 mạnh hơn, không yếu đi.

*(Có thể gộp vào RQ1 nếu cần xuống 4 RQ.)*

---

### RQ4. Mô hình có dự báo được — và *làm sao biết nếu không*?

*(gộp RQ5 + RQ6 + RQ6b cũ)* — **§VI**, dataset: **Alibaba v2021**

**(a) Signal gate — điều kiện tiên quyết, không phải câu hỏi riêng**

| | RCAEval | Alibaba 5h |
|---|---|---|
| $R^2$ trung vị CPU | **0,0034** | **0,3009** |
| % service $R^2>0{,}3$ | — | **50,1%** |
| $R^2$ trung vị Memory | 0,0034 | 0,0113 |
| Biên độ tải | 1,70 | 1,08 |

Khác **88×**. Benchmark RCA không dùng được cho capacity forecasting.

**(b) Dự báo, với giao thức có chế độ thất bại**

Negative control **PASS** (skill $-0{,}001$ … $-0{,}22$ khi hoán vị workload).

| model | mode | skill | %>0 |
|---|---|---|---|
| NNLS_deployed | in-dist | **+0,2719** | **86,9%** |
| NNLS_deployed | **OOD** | **+0,2684** | **69,5%** |
| LinearReg | OOD | +0,2854 | 66,7% |

$n = 1.293$. **OOD ≈ in-dist** → ngoại suy tốt gần bằng nội suy.

**(c) Dự báo Memory là artefact của metric**

| | MAPE | skill |
|---|---|---|
| Memory in-dist | **0,15** *(trông xuất sắc)* | **+0,006** |
| Memory OOD | 0,17 | **−0,05** |

Tái hiện trên **hai bộ dữ liệu độc lập** → tính chất của metric, không phải của
benchmark. Bản cũ trình bày $2$–$4\%$ MAPE như thành công → **phải rút**.

---

### RQ5. Cấu trúc nhân quả mua gì: *đồ thị*, toán tử $do$, hay không gì?

*(gộp RQ7 + RQ8 + RQ9 + RQ10 cũ)* — **§VII**, dataset: Alibaba + RCAEval

**(a) Đồ thị: CÓ giá trị** — Alibaba, negative control PASS

| | n | graph thắng | skill graph | skill uniform |
|---|---|---|---|---|
| in-dist | 27 | **23/27 (85,2%)** | **+0,4999** | +0,3081 |
| OOD | 27 | 17/27 (63,0%) | +0,4919 | +0,5660 |

Nội suy: thắng rõ. Ngoại suy: **đổi độ chính xác đỉnh lấy tính nhất quán**.
Sửa kết luận RQ4 cũ ($n{=}6$, CI chứa 0, không control).

**(b) Toán tử $do$: KHÔNG** — bốn phép kiểm độc lập

| Phép kiểm | n | Kết quả |
|---|---|---|
| Chuyển sang chế độ can thiệp | 1.080 | lợi thế là **ngoại suy**, không phải confounding |
| Confounder đồng-node | 3.780 | **không có** (p=0,546) |
| Confounder tải toàn cục | 2.340 | giảm tương quan dư **1,2%** |
| Backdoor adjustment | 1.080 | **không cải thiện** (51,2% = đồng xu) |
| Phản thực (abduction) | 3.240 | **quy về giữ phần dư** (47,5%) |

Cộng: `SCM_Deployed` **trùng khít** `LinearReg` trên **86/105** cặp, sole win
$0/21$ và $0/84$.

Trong cấu hình này — can thiệp tại node gốc, không confounder đo được —
$P(Y\mid do(X)) = P(Y\mid X)$ **đồng nhất**.

**(c) Ranh giới: khi nào $do$ *không* quy về điều kiện hoá**

$$\text{sai số tương đối} = \frac{\beta a_0}{d_0}\,\delta_A$$

Tới **57%** khi can thiệp đồng thời tổ tiên–hậu duệ. **Nhưng**: đây là *ngữ
nghĩa đúng*, không phải năng lực mà cài đặt cẩn thận thiếu. Đừng thổi phồng.

Ba câu trả lời phân cấp cho một câu hỏi — đọc thành lập luận, không thành bốn
thất bại rời rạc.

---

### Tiểu mục (KHÔNG phải RQ): hệ quả vận hành

**52,9%** khác verdict trên 136 lần chạy (47 requirement × 3), lệch mạnh nhất
dưới adversarial (69,6%). Là *khác*, **không phải** *đúng hơn* — không có ground
truth. Một con số, không đủ nặng làm RQ.

---

## §VIII Discussion — ba mục mới

1. **Hàm ý cho văn liệu causal-AIOps.** Nhiều công trình áp SCM vào đúng cấu
   hình mà nó tương đương hồi quy mà không nhận ra. Chúng tôi đo điều đó trên
   một hệ thật và cho thấy nó khiến một thiết kế trông hợp lý trở nên rỗng.

2. **Hàm ý cho benchmark.** RCA và capacity forecasting có **yêu cầu xung đột**
   về dữ liệu. Không benchmark nào phục vụ cả hai. Đây là lý do thứ hai, độc
   lập, khiến mảng ở lại phía hồi cứu.

3. **Hàm ý cho phương pháp đánh giá.** MAPE thấp trên biến phương sai nhỏ không
   phải bằng chứng năng lực. **Signal gate + baseline hằng số + negative
   control** nên là bắt buộc.

---

## Còn thiếu — phải xử lý trước khi nộp

| Việc | Bắt buộc? | Ghi chú |
|---|---|---|
| **Compile LaTeX** | ✅ chặn tất cả | `amsthm`+`algorithm` với IEEEtran chưa kiểm |
| **Backend LLM thứ 2–3** | ✅ | Groq hết quota ngày; **Claude Sonnet 5 ~$1,8 với caching** là rẻ nhất, và là bậc frontier |
| Đối chiếu 47 reference | ✅ | 33 cái viết từ trí nhớ, số trang chưa kiểm |
| Tên tác giả thật | ✅ | còn placeholder |
| Benchmark parser đầy đủ trên Train Ticket | ⚠️ nên có | code đã chạy, chưa chạy bộ 50 prompt |
| Mở rộng 27 edge Tier-1 | ⚠️ | nút cổ chai là **giao tập service** CallGraph↔MCR, không phải thời gian |
| Threshold-crossing F1 trên Alibaba | ⚠️ | cần ngưỡng đại diện (percentile) |

---

## Những gì **phải rút** khỏi bản cũ

- Mọi tuyên bố độ chính xác dự báo trên RCAEval, **kể cả Memory 2–4%**
- Table I và II ở vai trò hiện tại (chuyển thành bối cảnh cho signal gate)
- Backpressure extension như "cải tiến 23/23 node" → viết lại thành **case
  study cảnh báo** (RQ7 đo được cái giá: +365 MAPE khi cha ra ngoài miền)
- Tuyên bố $do(x)$ là đóng góp trung tâm
- GMR $0\%$ như tác dụng của guard (confounded)
