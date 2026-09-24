# Bộ khung RQ v3 — "Admissible Evaluation"

> ⚠️ **ĐÃ LỖI THỜI — `docs/paper_draft.tex` là nguồn đúng.** Tài liệu này mô tả khung V3 với
> 5 RQ **trước** hai thay đổi đã áp vào bản thảo (2026‑09‑23):
> 1. **RQ2 Attainability + RQ3 Realisation đã GỘP** thành một RQ2 "Guarantee" — chúng là cùng
>    một mệnh đề ở hai hiện thân (bảo đảm được công bố, nhưng đánh giá thiết lập nó không có
>    khả năng bác bỏ nó), khác nhau chỉ ở chỗ đóng không gian kết cục: cỡ mẫu hay topology.
> 2. **Thêm RQ5 "Prospective Validation"** — không có trong tài liệu này vì khi viết nó,
>    testbed `SS-LIMITS*` chưa tồn tại.
>
> Khung đúng hiện tại: **RQ1 Necessity · RQ2 Guarantee · RQ3 Measurability · RQ4 Attribution ·
> RQ5 Prospective**. Số claim sống sót giờ là **hai**, không phải một. Xem bảng dịch ở đầu
> `README.md` gốc.

Tài liệu này định nghĩa lại 5 RQ của bài từ **một nguyên lý duy nhất**, thay vì
năm chủ đề rời rạc. Không yêu cầu thí nghiệm mới để *phát biểu*; phần cuối liệt
kê chính xác chỗ nào cần tăng `n` để *chống đỡ* được ở mức Q1.

---

## 1. Nguyên lý thống nhất

> **Definition 1 (Admissible evaluation).** Let $C$ be a claim about a pipeline
> component and $E$ an evaluation procedure with outcome space $O$. $E$ is
> **admissible** for $C$ if there exists $o \in O$ whose observation refutes $C$;
> otherwise $E$ is **vacuous** for $C$. Admissibility is a property of the pair
> $(E, C)$: the same metric may be admissible for one claim and vacuous for
> another.

**Luận điểm trung tâm của bài:** trong một pipeline có LLM đứng ở biên, đánh giá
*thông lệ* của mỗi tầng là **vacuous** đối với chính tính chất chi phối độ tin cậy
của tầng đó. Với mỗi tầng, bài chỉ ra (i) mệnh đề chi phối, (ii) đánh giá thông
lệ và lý do nó không thể trả lời "không", (iii) đánh giá khả bác thay thế, và
(iv) kết cục bác bỏ mà đánh giá đó cho phép xảy ra.

**Câu chốt dùng cho Abstract/Conclusion:**

> Of the five claims we set out to defend, one survived intact. The evaluation
> design that allowed the other four to fail — three of them against results we
> had previously reported ourselves — is the paper's transferable content.

Đây là thứ biến 5 kết quả rời rạc thành 5 hiện thân của một định luật, và biến
các kết quả âm từ *lời xin lỗi* thành *đóng góp*.

---

## 2. Khuôn mẫu chung cho mọi RQ

Mỗi RQ **bắt buộc** khai báo đủ 7 mục. Đây là phần "chặt chẽ": reviewer không thể
hỏi "làm sao biết các anh sai" vì mục 5 đã trả lời trước.

| # | Mục | Vai trò |
|---|---|---|
| 1 | **Claim** $C_i$ | Phát biểu sao cho *có thể sai* |
| 2 | **Incumbent** | Đánh giá thông lệ của cộng đồng |
| 3 | **Vacuity** | Chứng minh incumbent không thể trả "không" cho $C_i$ |
| 4 | **Admissible design** | Đánh giá thay thế |
| 5 | **Refuting outcome** | Kết cục cụ thể sẽ bác bỏ $C_i$ |
| 6 | **Finding** | Kết quả + $n$ |
| 7 | **Power** | Mẫu hiện có đủ hay thiếu, thiếu thì cần gì |

---

## 3. Năm RQ

Trục xương sống, mỗi RQ một từ khoá: **Necessity → Attainability → Realisation →
Measurability → Attribution.** Hai RQ đầu hỏi *verifier có cần và bảo đảm được gì*;
RQ3 hỏi *hiện thực có đúng đặc tả*; hai RQ cuối hỏi *mô hình hạ nguồn có nội dung gì*.

### RQ1 — Necessity

> **Is the verifier necessary, or does generator capability subsume it?**

| | |
|---|---|
| **Claim** | $C_1$: structural violation rate is a decreasing function of generator capability — hence the verifier is transitional, obsoleted by the next model. |
| **Incumbent** | Average-case task accuracy on a single backend; or the guarded-vs-unguarded outcome contrast on PBVR/SHR/GMR. |
| **Vacuity** | Hai lý do độc lập. (a) Một chỉ số gộp không tách được lỗi *ngữ nghĩa* khỏi lỗi *cấu trúc*, nên không có kết cục nào của nó bác bỏ được $C_1$. (b) Trên mọi cấu hình có guard, PBVR và SHR **bằng 0 theo Corollary 1** với mọi backend và mọi prompt — bảng kết quả là tautology, không quan sát nào làm nó khác đi. |
| **Admissible design** | Giữ cố định task, prompt, harness; **chỉ thay generator** qua các bậc năng lực. Báo cáo các chế độ lỗi **tách rời** (PBVR / SHR / GMR / Anchor MAE) và **theo từng nhóm prompt**. Bổ sung **firing rate** của từng layer trên input lành tính và input đối kháng — *firing rate, chứ không phải outcome rate*, mới là đại lượng phân biệt một bảo đảm đang hoạt động với một bảo đảm trang trí. |
| **Refuting outcome** | GMR và SHR giảm theo năng lực với **cùng nhịp** mà Anchor MAE giảm ⇒ $C_1$ đứng vững, verifier chỉ là giải pháp tình thế. Hoặc: một layer **không bao giờ kích hoạt** kể cả dưới input đối kháng ⇒ layer đó vô dụng. |
| **Finding** | Anchor MAE giảm $25\times$ ($844.6\% \to 33.6\%$) trong khi GMR $100 \to 92\%$, SHR $91.3 \to 86\%$; GMR theo nhóm 80–100%, không tập trung ở nhóm đối kháng. Layer 2 kích hoạt 18/18 đối kháng, 0/150 lành tính; Layer 3 33/36 đối kháng, 0 lành tính (kể cả khi gỡ chỉ dẫn khỏi prompt — loại trừ tính vòng quanh). |
| **Power** | ⚠️ **Mắt xích yếu nhất bài.** $n = 2$ generator, arm thứ hai chỉ 1 repeat. Bộ đối kháng 18 + 12 prompt do chính tác giả dựng. |

---

### RQ2 — Attainability

> **Where no total projection exists, what guarantee is attainable, and what does it cost?**

| | |
|---|---|
| **Claim** | $C_2$: the abstention layer certifies an out-of-scope miss rate of $\alpha$. |
| **Incumbent** | Công bố $\alpha$ danh nghĩa của thủ tục conformal. |
| **Vacuity** | $\alpha$ là **con số được chọn**, không phải được đo. Không quan sát nào trên tập calibration bác bỏ được một con số do ta tự đặt. Đây là dạng vacuity tinh vi nhất trong bài: nó **trông giống** một bảo đảm thống kê. |
| **Admissible design** | Công bố **thống kê thứ tự mà cỡ mẫu thực sự chọn ra**, chứ không phải mức tin cậy mong muốn: split conformal đặt ngưỡng tại thống kê thứ tự thứ $\lceil (n{+}1)\alpha \rceil$, nên mọi $\alpha < 1/(n{+}1)$ chỉ chọn lại đúng thống kê cực biên. Kèm theo: chi phí **coverage** tại điểm vận hành. |
| **Refuting outcome** | Ngưỡng **bất biến theo $\alpha$** — chẩn đoán trực tiếp được: nếu $\alpha = 0.05$ và $\alpha = 0.01$ trả về cùng một ngưỡng thì mức danh nghĩa không chứng nhận gì. |
| **Finding** | Tại $n_{\text{pos}} = 11$, cả hai $\alpha$ trả về **đúng cùng ngưỡng $0.5343$**. Cận khả đạt là $1/22 = 4.5\%$, nên $\alpha = 0.01$ ta từng dùng là bất khả đạt. Chi phí: 19.8% chuyển người xét, 5.0% từ chối nhầm, 80.7% cuộc gọi nhận được con số. |
| **Power** | ⚠️ Ngưỡng được **fit và đánh giá trên cùng 81 điểm**; logistic score fit trên 61 trong khi ngưỡng dùng 81 ⇒ exchangeability chỉ gần đúng, phát biểu conformal phải đọc là *design target*, không phải *certified bound*. |

> **Ghi chú quan trọng:** đây chính là lý do RQ2 hiện tại "trông như mục
> Limitations". Dưới khung mới nó **không còn là** limitation — nó là một hiện
> thân hợp lệ của nguyên lý: *một mức tin cậy danh nghĩa là bảo đảm bất khả bác;
> mức khả đạt do cỡ mẫu quyết định.* Đó là bài học chuyển giao được cho mọi bài
> dùng conformal trên tập calibration nhỏ — và có rất nhiều bài như vậy.

---

### RQ3 — Realisation

> **Does the implementation satisfy its own proposition on a system it was not built on?**

| | |
|---|---|
| **Claim** | $C_3$: the implementation realises $\Pi$, hence Corollary 1 holds in deployment. |
| **Incumbent** | Test suite + benchmark trên chính topology phát triển. |
| **Vacuity** | Các invariant gắn với **thuộc tính ngẫu nhiên** của topology đó — đúng một gateway; tồn tại một node tên đúng là `front-end` — **không thể bị vi phạm ở đó**. Không phải test yếu; là không gian kết cục không chứa kết cục bác bỏ. |
| **Admissible design** | Triển khai sang topology khác ở đúng những thuộc tính ngẫu nhiên đó: 68 vs 7 node, 14 vs 1 gateway, từ vựng tên rời nhau, miền nghiệp vụ khác. |
| **Refuting outcome** | Bất kỳ intervention phát ra nào vi phạm Prop. 1(i)–(iii). |
| **Finding** | Ba defect. Defect 3 là phát hiện thực chất: bộ lọc service khi loại hết đề xuất đã trả về hằng `['front-end']` — trên hệ không có node tên đó, **guard tự chèn vào một service không tồn tại**, vi phạm đúng Prop. 1(iii) mà nó sinh ra để thực thi, và làm sai Corollary 1 ngoài Sock Shop. |
| **Power** | ⚠️ $n = 1$ hệ thống bổ sung, phát hiện **cơ hội** chứ không hệ thống. Reviewer đọc ra là war story. |

> **Nâng cấp bắt buộc để RQ3 có sức nặng:** biến từ giai thoại thành **phương
> pháp**. Sinh $N$ topology tổng hợp tham số hoá theo (số gateway, out-degree của
> gateway, độ sâu, từ vựng tên), chạy verifier trên toàn bộ, và báo cáo *invariant
> nào vỡ như một hàm của thuộc tính topology nào*. Không tốn LLM. Khi đó defect 3
> là **một ca do phương pháp tự tìm ra**, mạnh hơn hẳn việc kể lại rằng nó tình cờ lộ.

---

### RQ4 — Measurability

> **Can the data support a forecasting claim at all, and where it can, does the model forecast?**

| | |
|---|---|
| **Claim** | $C_4$: the capacity model forecasts resource consumption from workload. |
| **Incumbent** | MAPE trên telemetry held-out. |
| **Vacuity** | Một target **gần như bất biến** cho MAPE thấp dưới *mọi* predictor, kể cả predictor bỏ qua hoàn toàn đầu vào. Không có ngưỡng MAPE nào phân biệt được "học được cơ chế" với "target không đổi" — đúng như kết quả memory 2–4% của chính chúng tôi. |
| **Admissible design** | Cổng bốn tầng, **mỗi tầng đều có thể trả về kết quả âm**: (1) *signal gate* — $R^2$ của predictor với target trên dữ liệu **chưa chia**; (2) *baseline bỏ qua predictor* — hằng số dự đoán trung bình tập huấn luyện; (3) *negative control* — hoán vị cột workload trong từng service, phá quan hệ nhưng giữ phân phối biên; (4) *skill* thay cho sai số thô: $1 - \mathrm{MSE}_{\text{model}}/\mathrm{MSE}_{\text{const}}$. |
| **Refuting outcome** | $R^2 \approx 0$ ở cổng ⇒ **không claim nào về độ chính xác là khả bác trên dữ liệu này**; hoặc skill $\le 0$ ⇒ không học được gì; hoặc skill **sống sót** qua hoán vị ⇒ chỉ số đang đo thứ khác. |
| **Finding** | RCAEval $R^2 = 0.0034$ ⇒ benchmark **bất khả dụng** cho câu hỏi này, và điều đó giải thích trong một chẩn đoán *mọi* kết quả âm ở RQ5. Alibaba $R^2 = 0.3009$ ($88\times$). CPU skill $+0.27$ trong phân phối và $+0.27$ ngoài phân phối, thắng hằng số trên 86.9% / 69.5% của 1 293 service. Memory skill $+0.006$ ⇒ **rút lại** claim. Negative control đạt: mọi model sụp về $\le 0$. |
| **Power** | ✅ Mẫu mạnh nhất bài ($n = 1\,293$). ⚠️ Nhưng claim *"benchmark RCA không đỡ được capacity forecasting"* lại chỉ dựa trên **một họ benchmark**. |

> **Nâng cấp:** chạy signal gate trên nhiều dataset (RCAEval các biến thể,
> DeathStarBench tự sinh, Alibaba 2022, trace Azure/Borg) để biến nó thành một
> **tiêu chí admissibility tái sử dụng được** mà người khác chạy được trên dữ liệu
> của họ. Đó là đóng góp có tuổi thọ, khác hẳn một nhận xét về một benchmark.

---

### RQ5 — Attribution

> **Which causal commitment survives an evaluation able to reject it — the graph, the $do$-operator, or neither?**

| | |
|---|---|
| **Claim** | $C_{5a}$: propagation along the real dependency graph beats a uniform-delta assumption. $C_{5b}$: the $do$-operator confers an estimation advantage over conditioning. |
| **Incumbent** | Độ chính xác held-out quan sát, **kể cả** held-out OOD theo tải. |
| **Vacuity** | Vacuous đối với *interventional transfer*: một cơ chế có thể cải thiện sai số held-out trên **mọi** node mà vẫn đổ vỡ dưới can thiệp, ngay khi một parent rời khỏi khoảng quan sát. Đây không phải giả định — mở rộng backpressure của chính chúng tôi đã làm đúng vậy: tốt lên trên 23/23 node held-out, rồi **tệ đi 365 điểm MAPE** dưới can thiệp thật. |
| **Admissible design** | Ground truth can thiệp thật (fault injection), phân tầng theo *parent còn trong khoảng huấn luyện hay không*; backdoor vs naive trên interventional transfer; abduction vs cộng thẳng residual; và kiểm đếm số cặp NNLS/OLS **dự đoán trùng khít** (nơi phép kiểm định ghép cặp không xác định). |
| **Refuting outcome** | $C_{5b}$ bị bác nếu backdoor adjustment không thắng được estimator naive. $C_{5a}$ bị bác nếu graph không thắng uniform-delta trên đa số cạnh. |
| **Finding** | $C_{5a}$ **sống**: 23/27 cạnh (85.2%), skill trung vị $+0.50$ vs $+0.31$; OOD graph thắng trên nhiều cạnh hơn (17/27 vs 11/27) dù skill trung vị thấp hơn — đánh đổi đỉnh lấy tính nhất quán. $C_{5b}$ **bị bác**: backdoor thắng 553/1080 (tung đồng xu), hiệu trung vị $0.000$; không đo được confounder nào ($p = 0.546$); counterfactual abduction không thêm gì so với cộng residual. Lý do là **cấu trúc**: can thiệp tại gateway nằm ở node in-degree 0, nên truncated factorisation **không cắt cạnh nào** và $P(Y \mid do(X)) = P(Y \mid X)$ đồng nhất. |
| **Power** | ⚠️ 27 cạnh cho claim chủ lực $C_{5a}$. Lập luận cấu trúc cho $C_{5b}$ đúng nhưng đang chỉ nói về hệ của chính mình. |

> **Nâng cấp:** tổng quát hoá $C_{5b}$ khỏi hệ của mình bằng một **audit tài liệu**:
> lấy 12–20 bài causal-microservice, phân loại **vị trí can thiệp** của từng bài,
> chỉ ra bài nào áp dụng $do()$ ở nơi chứng minh được là tương đương conditioning.
> Biến một sự thật sách giáo khoa thành một phát hiện về cả dòng nghiên cứu.

---

## 4. Bảng ánh xạ khung cũ → khung mới

| Khung cũ | Khung mới | Thay đổi thực chất |
|---|---|---|
| RQ1 "provable **and** load-bearing" | **RQ1 Necessity** | Bỏ liên từ trong tiêu đề; "load-bearing" thành *firing rate* — mục 4 của khuôn mẫu, không còn là nửa câu hỏi thứ hai |
| §rq1-backend (tiểu mục) | Vẫn là tiểu mục của RQ1 | Nhưng giờ là **thiết kế khả bác** của RQ1, không phải một phép kiểm tra thêm |
| RQ2 "abstention cost" | **RQ2 Attainability** | Từ *limitation* thành *hiện thân của nguyên lý*: $\alpha$ danh nghĩa là bảo đảm vacuous |
| RQ3 "transfer + what it reveals" | **RQ3 Realisation** | Từ *war story* thành *kiểm chứng hiện thực đối chiếu đặc tả* |
| RQ4 "forecast + how would we know" | **RQ4 Measurability** | Bỏ vế tu từ khỏi tiêu đề; đưa vào mục *Refuting outcome* |
| RQ5 "graph, do-operator, or neither" | **RQ5 Attribution** | Giữ; thêm tường minh incumbent vacuous (held-out quan sát) |

**Label `sec:rq1..rq5` giữ nguyên** — không phải sửa tham chiếu chéo nào.

---

## 5. Việc cần làm, xếp theo chi phí/lợi ích

| # | Việc | Chi phí | Tác động |
|---|---|---|---|
| 1 | **Proposition 2 — chứng chỉ khoảng end-to-end.** Mọi cơ chế Tier-1/Tier-2 là ánh xạ tuyến tính không âm ($\beta \ge 0$); latency dùng $\phi(w) = w/(c-w)$ đơn điệu tăng trên $w < c$. Hợp thành đơn điệu trên DAG vẫn đơn điệu ⇒ với $\delta \in [5,50]$ do $\Pi$ bảo đảm, mọi node hạ nguồn nằm trong $[F(5), F(50)]$, tính được bằng **hai lượt forward**. | ~0 thí nghiệm | **Rất cao.** Nối Prop. 1 với con số kỹ sư thực sự đọc; pipeline xuất *khoảng có chứng chỉ* thay vì điểm; và biến ràng buộc NNLS từ một **khoản chi** ($+0.2719$ vs $+0.2830$) thành thứ **mua được** tính hợp lệ của chứng chỉ. Cần xử lý tường minh nhiễu $N$: chiếu theo kỳ vọng hoặc dùng phân vị. |
| 2 | **Backend sweep**: ≥5 bậc năng lực, **bắt buộc có reasoning model tuyến đầu**, ≥3 repeat mỗi model | Trung bình (quota API) | **Rất cao.** Nếu GMR vẫn cao ở model tuyến đầu, $C_1$ bị bác không thể chối cãi và riêng RQ1 đủ sức nặng Q1. Nếu GMR sụp — vẫn tốt, ta biết trước reviewer |
| 3 | **Property-based testing trên topology tổng hợp** (RQ3) | Thấp, không tốn LLM | Cao — biến giai thoại thành phương pháp |
| 4 | **Signal gate đa dataset** (RQ4) | Trung bình | Cao — thành tiêu chí tái sử dụng được |
| 5 | **Audit vị trí can thiệp trong tài liệu** (RQ5) | Thấp, chỉ đọc | Cao — tổng quát hoá khỏi hệ của mình |
| 6 | **Tập prompt do người ngoài nhóm gán nhãn** + kiểm tra held-out cho Scope Gate | Trung bình | Trung bình — vá điểm yếu "một tác giả taxonomy" |

**Không khuyến nghị:** user study. Đắt, và không phải chỗ bài đang yếu.

---

## 6. Trạng thái áp dụng vào `paper_draft.tex`

| Phần | Trạng thái |
|---|---|
| Title | ✅ đổi sang *Admissible Evaluation: What Five Standard Metrics Cannot Refute in an LLM-Fronted Analysis Pipeline* |
| Abstract | ✅ viết lại (286 từ — vẫn trên mức 250 của IEEE TSE, cần cắt thêm một mệnh đề khi nộp) |
| Introduction | ✅ viết lại; **Definition 1 đặt ở đây**, không phải Section V |
| Contributions (5 mục) | ✅ viết lại, ánh xạ 1–1 với khung: nguyên lý → RQ1 → RQ2+RQ3 → RQ4 → RQ5 |
| Section V mở đầu + `tab:rq-spine` | ✅ |
| 5 tiêu đề RQ + dòng *Claim / Refuted by* | ✅ |
| Conclusion | ✅ viết lại: 1 sống sót → 4 bác bỏ (mỗi cái nêu một kiểu đóng không gian kết cục) → tổng quát hoá |
| Related Work §II-A | ⬜ vẫn đóng khung theo "necessity of the verifier", nên nối lại với admissibility |
| Threats to Validity | ⬜ vẫn theo Wohlin; cân nhắc thêm một đoạn phân biệt *vacuity* với *threat* |

**Tồn tại chưa sửa (có từ trước):** ba bảng chưa được tham chiếu trong thân bài —
`tab:dataset-gap`, `tab:rq1-backend`, `tab:rq5-graph`. IEEE yêu cầu mọi bảng phải
được gọi tên trong văn bản.
