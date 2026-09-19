# Kịch bản thuyết trình — *Capability Is Not Verification*

**Bối cảnh:** Seminar nhóm nghiên cứu / lớp học, 20–30 phút + thảo luận.
**Nguồn:** `docs/paper_draft.tex`.
**Cách dùng:** Mỗi mục = 1 slide. Dòng **[SLIDE]** là nội dung nên hiện trên slide (ngắn, ít chữ). Phần văn xuôi bên dưới là *lời nói* — không đọc y nguyên, chỉ để nắm ý và số liệu chính xác. Tổng thời lượng ước tính ghi ở mỗi slide; cộng lại ≈ 24 phút nói, còn lại cho chuyển tiếp/Q&A.

---

## 0. Trang bìa (0:30)

**[SLIDE]**
> Capability Is Not Verification: Provable Constraints on LLM-Translated Requirements for Microservice Capacity Analysis
> — Tên bạn, đơn vị

**Nói:** Chào hội đồng/nhóm. Đề tài của em trả lời một câu hỏi rất cụ thể: khi một LLM được đặt ở đầu một pipeline phân tích — dịch một yêu cầu bằng ngôn ngữ tự nhiên thành đầu vào có cấu trúc cho hệ thống downstream — thì ta *đảm bảo* được gì về tính hợp lệ của đầu ra đó, chứ không chỉ đo được độ chính xác trung bình của nó.

---

## 1. Vấn đề: pipeline bị chi phối bởi output tệ nhất, không phải trung bình (2:00)

**[SLIDE]**
- LLM ngày càng được đặt ở *ranh giới pipeline*: dịch văn bản → object có cấu trúc → hệ thống downstream (compiler/solver/simulator) tiêu thụ như thể luôn hợp lệ
- Độ tin cậy của pipeline bị chi phối bởi **output tệ nhất**, không phải trung bình
- 1 object sai cấu trúc có thể lan thành 1 con số "chính xác" mà kỹ sư hành động theo

**Nói:** Cách đánh giá phổ biến hiện nay cho các hệ thống "LLM-as-translator" là accuracy trung bình trên một tập test. Nhưng nếu LLM là một khâu trong pipeline tự động — không có người đọc lại đầu ra — thì rủi ro thực sự nằm ở *worst-case*, không phải mean-case. Một đầu ra sai cấu trúc (ví dụ trỏ sai node, đề xuất một mức tải phi vật lý) vẫn có thể được các bước sau xử lý như dữ liệu hợp lệ, và sinh ra một con số nhìn rất thuyết phục nhưng vô nghĩa.

---

## 2. Giả định thường gặp — và vì sao nó sai (2:00)

**[SLIDE]**
- Phản xạ thường thấy: "output sai → dùng model to hơn / prompt tốt hơn"
- Giả định ngầm: **validity là hệ quả của capability**
- Bài báo kiểm định giả định này bằng thực nghiệm — và nó **không đúng theo cách quan trọng**

**Nói:** Phản ứng tự nhiên khi thấy LLM sai là cải thiện generator. Giả định ngầm ở đây là: model càng mạnh thì đầu ra càng hợp lệ. Đóng góp trung tâm của bài báo là cho thấy capability và validity là hai trục **tách biệt** — model mạnh hơn cải thiện được sự *hợp lý về ngữ nghĩa* (magnitude nhìn có vẻ đúng) nhưng **không** cải thiện tương ứng *tính hợp lệ cấu trúc* (đúng vị trí, đúng ràng buộc vật lý). Em sẽ cho số liệu cụ thể ở phần RQ1.

---

## 3. Bài toán cụ thể: dự phóng tải cho một tính năng *chưa tồn tại* (2:00)

**[SLIDE]**
- Input: yêu cầu tính năng bằng ngôn ngữ tự nhiên (chưa được triển khai)
- Output: can thiệp $do(X{=}x)$ có giới hạn trên đồ thị phụ thuộc microservice
- Mục tiêu: dự phóng tải trước khi bỏ công sức triển khai → đánh giá *feasibility*
- Vì sao chọn bài toán này: **validity ở đây là decidable** — kiểm tra được bằng vị từ cụ thể (đúng service tồn tại? đúng gateway? đúng khoảng vật lý cho phép?)

**Nói:** Bài toán chọn: cho một yêu cầu tính năng mới, dịch nó thành một can thiệp có giới hạn trên đồ thị kiến trúc microservice, rồi mô phỏng lan truyền tải để trả lời "tính năng này có khả thi về mặt tài nguyên không, trước khi build". Điểm quan trọng về mặt phương pháp luận: bài toán này được chọn *vì* tính hợp lệ của một can thiệp là quyết định được bằng các vị từ kiểm tra được — tên service có tồn tại trong kiến trúc không, điểm tiêm có phải gateway không, độ lớn có nằm trong khoảng vật lý cho phép không. Điều đó cho phép **chứng minh** một verifier là sound, thay vì chỉ đo nó trên tập test.

---

## 4. Năm đóng góp chính (2:30)

**[SLIDE]**
1. Verifier có **chứng minh toán học** (Proposition — total, deterministic), và chứng minh nó *load-bearing* bằng adversarial test
2. Bằng chứng: **capability không thay thế được verification** (model 117B vs. model nhỏ)
3. Bằng chứng: **một thiết kế đã chứng minh vẫn cần validate lại khi triển khai** ở hệ thống khác — phát hiện lỗi vi phạm chính proposition
4. Một **protocol đánh giá có thể trả lời "sai"** — và dùng nó để chỉ ra benchmark RCA chuẩn không đánh giá được capacity forecasting
5. Tường minh **rút lại** một kết quả cũ (memory forecasting) khi phát hiện nó là artefact của metric

**Nói:** Đây là 5 đóng góp, em sẽ đi qua từng cái theo cấu trúc: kiến trúc hệ thống → chứng minh → thực nghiệm RQ1–RQ5 → thảo luận. Điểm em muốn nhấn mạnh ngay từ đầu: đóng góp số 5 hơi khác thường — đây là một sự rút lại (retraction) kết quả cũ của chính nhóm, không phải một kết quả dương tính. Em nghĩ đó cũng là một đóng góp có giá trị, vì nó cho thấy protocol đánh giá mới *có khả năng bắt được* lỗi mà protocol cũ bỏ lọt.

---

## 5. Kiến trúc hệ thống tổng quan (2:00)

**[SLIDE]** *(vẽ sơ đồ 4 node)*
1. **Parser Agent** — text → $do(X{=}x)$ có giới hạn (3 lớp kiểm chứng)
2. **Architecture Agent** — BFS từ gateway → "blast radius" các service bị ảnh hưởng
3. **Capacity Agent** — lan truyền tải 2 đường: SCM song biến nhanh + DAG toàn cục chính xác
4. **Report node** — LLM tổng hợp thành `FeasibilityReport` ngôn ngữ tự nhiên

**Nói:** Hệ thống là một state machine 4 node chạy tuần tự (LangGraph). Node 1 là nơi có "phép màu" LLM và cũng là nơi rủi ro nằm — em sẽ đào sâu ngay sau đây. Node 2 chỉ là graph traversal tất định. Node 3 là mô hình lan truyền tải nhân quả — phần này được đánh giá ở RQ4–RQ5. Node 4 tổng hợp báo cáo, không được đánh giá định lượng trong bài.

---

## 6. Lớp kiểm chứng — vì sao cần 3 lớp, không phải prompt tốt hơn (2:30)

**[SLIDE]** *(hiện sơ đồ Fig. verification: candidate → L1 → L2 → L3 → output)*
- **Nguyên tắc LLM-Modulo:** LLM = generator không tất định; verifier tất định, độc lập với LLM
- **Layer 1 — Scope Gate**: *selective abstention* — điểm $s(x)$ so với 2 ngưỡng conformal → pass / `NEEDS_HUMAN_REVIEW` / refuse hẳn
- **Layer 2 — Entity Grounding**: mọi entity phải tồn tại trong đồ thị; **Gateway Invariance** — điểm tiêm bắt buộc tại node in-degree = 0, nếu sai thì fallback về gateway gần nhất
- **Layer 3 — Bounded Projection**: clamp độ lớn can thiệp vào khoảng vật lý cho phép $[5,50]$

**Nói:** Ba lớp có 3 mục đích toán học khác nhau: abstention (từ chối khi không đủ tin cậy), graph-membership (ép buộc thuộc tính cấu trúc), và projection khoảng giá trị. Layer 1 khác 2 lớp còn lại về bản chất — nó có thể *từ chối trả lời* thay vì sửa. Layer 2 và 3 luôn trả về một candidate hợp lệ, không bao giờ từ chối — đây là 2 lớp được chứng minh sound.

---

## 7. Định lý trung tâm: Proposition (Soundness) (2:00)

**[SLIDE]**
- $\Pi$ = Layer 2 $\circ$ Layer 3 là hàm **tổng (total) và tất định (deterministic)**
- Với **mọi** candidate $c = L(r)$ do LLM sinh ra — bất kể adversarial, bất kể LLM có ảo giác hay không —  $\Pi(c)$ luôn thỏa:
  (i) điểm tiêm tại gateway hợp lệ, (ii) $\delta' \in [5,50]$, (iii) $S' \subseteq V$
- Chứng minh **không phụ thuộc phân phối của $L$** — không cần LLM "đủ tốt"

**Nói:** Đây là khác biệt cốt lõi so với benchmark accuracy thông thường: đây không phải là "chúng tôi đo thấy 95% output hợp lệ trên tập test", mà là một chứng minh — *mọi* input, kể cả input ác ý chưa từng thấy, đều cho ra output hợp lệ về cấu trúc. Điều đó có nghĩa: guarantee này không suy yếu khi gặp adversarial input ngoài tập test, vì nó không được suy ra từ tập test.

---

## 8. Mô hình lan truyền tải — Capacity Agent (1:30)

**[SLIDE]**
- Đồ thị 2 tầng: $V = V_W \cup V_R$ (workload nodes ↔ resource nodes)
- Tầng 1 (workload↔workload): lan truyền chéo dịch vụ theo call-graph
- Tầng 2 (workload→resource): chuyển hóa tải thành tiêu thụ tài nguyên tại từng service
- Cơ chế NNLS (non-negative least squares), giữ tính đơn điệu (monotonicity)

**Nói:** Đây là phần mô hình nhân quả — không đi sâu công thức ở đây, chỉ cần nắm ý: tải lan truyền theo 2 tầng tách biệt, cho phép mô hình vừa nắm được sự phụ thuộc thực giữa các service, vừa đảm bảo tính đơn điệu vật lý (tải tăng thì tài nguyên không giảm). Phần này được đánh giá kỹ ở RQ4 và RQ5.

---

## 9. RQ1 — Capability không thay thế Verification (2:30)

**[SLIDE]** *(bảng số)*
| | Model nhỏ (unguarded) | Model 117B (unguarded) |
|---|---|---|
| Sai số biên độ (Anchor MAE) | $844.6\%$ | $33.6\%$ ($25\times$ tốt hơn) |
| Gateway misdirection | $100\%$ | $92\%$ |
| Service hallucination | $91.3\%$ | $86\%$ |
| **Với verifier (guarded)** | GMR $=0\%$, PBVR $=0\%$ | — |

**Nói:** Đây là kết quả trung tâm của bài. Khi thay một model lite-tier bằng một model open-weight 117B — lớn hơn nhiều — sai số về *độ lớn* con số giảm 25 lần. Nhưng tỉ lệ đề xuất sai gateway và hallucinate service gần như không đổi. Nói cách khác: model to hơn giỏi "đoán con số nghe hợp lý" hơn, nhưng **không** giỏi hơn trong việc tôn trọng ràng buộc cấu trúc của hệ thống thật. Khi bật verifier, GMR và PBVR về 0% — không phải vì generator giỏi hơn, mà vì lớp 2/3 chặn đứng sai số đó một cách tất định. Đó chính là bằng chứng thực nghiệm cho luận điểm ở slide 2.

---

## 10. RQ2 — Cái giá của việc "từ chối trả lời", và giới hạn của guarantee (2:00)

**[SLIDE]**
- ~1/5 request nhận **không phải một con số**: $12.0\%$ `NEEDS_HUMAN_REVIEW`, $7.3\%$ refuse
- Guarantee thống kê của Scope Gate bị giới hạn bởi **cỡ mẫu calibration** ($n=21$), không phải bởi $\alpha$ ta chọn
  → miss-rate chặt nhất chứng minh được là $1/22 = 4.5\%$, không phải $\alpha=1\%$ như kỳ vọng ban đầu
- Adversarial set lộ điểm yếu của phiên bản *trước*: keyword-overlap đơn thuần bị qua mặt $30/30$ lần

**Nói:** Layer 1 không "miễn phí" — nó đánh đổi coverage lấy độ an toàn. Khoảng 1 trong 5 yêu cầu không nhận được một con số, mà nhận review hoặc từ chối. Và điểm em muốn nhấn ở đây là tính trung thực về mặt thống kê: guarantee đạt được bị chặn bởi cỡ mẫu calibration, chứ không phải do ta chọn ngưỡng tin cậy tùy ý — đây là một ràng buộc toán học của conformal prediction, không phải lựa chọn thiết kế.

---

## 11. RQ3 — Chuyển sang hệ thống thứ hai: chứng minh không thay thế được triển khai đúng (2:00)

**[SLIDE]**
- Deploy verifier lên **Train Ticket** (68 nodes, 14 gateway) thay vì Sock Shop (7 nodes, 1 gateway)
- Phát hiện **3 lỗi vô hình** trên topology gốc — trong đó có **1 lỗi vi phạm chính Proposition đã chứng minh** (hard-coded fallback)
- Nguyên nhân: topology gốc chỉ có 1 gateway → mọi đường fallback đều trả về đúng node đó, lỗi không thể lộ ra

**Nói:** Đây là phần em cho là thú vị nhất về mặt phương pháp luận: một chứng minh toán học đúng *trên giấy* vẫn có thể sai *trong code* — và một hệ thống với chỉ 1 gateway không đủ để phát hiện việc này, vì mọi nhánh xử lý lỗi tình cờ đều dẫn về cùng 1 điểm. Đưa hệ thống sang một topology có 14 gateway đã "ép" các nhánh code khác nhau phải thực thi, và lộ ra lỗi. Bài học: **proof ≠ validated implementation**, kiểm chứng chéo trên topology khác là cần thiết.

---

## 12. RQ4 — Một protocol đánh giá *có thể* trả lời "sai" (2:30)

**[SLIDE]**
- 4 lớp bảo vệ: (1) signal gate $R^2$, (2) baseline hằng số, (3) negative control (permute workload), (4) so sánh in/out-of-distribution
- **Benchmark RCA chuẩn không đo được capacity forecasting**: $R^2 = 0.003$ (so với $0.30$ trên production trace)
- Trên dữ liệu có tín hiệu thật: CPU forecast skill $+0.27$ in-distribution, **$+0.27$ out-of-distribution** (ngoại suy tốt ngang nội suy)
- **Rút lại** kết quả cũ: memory MAPE $2$–$4\%$ — hóa ra baseline hằng số cũng đạt được, skill $\approx 0$

**Nói:** Đây là phần "tự phê bình" có chủ đích của bài báo. Lần đầu đo, nhóm báo cáo memory forecast với MAPE 2-4% và tưởng đó là thành công — nhưng vì memory gần như bất biến theo tải, một baseline không học gì cả cũng đạt sai số y hệt. Protocol mới thêm 4 lớp kiểm tra để *bắt* chính xác lỗi này trước khi công bố, thay vì sau khi phản biện chỉ ra. Với CPU, tín hiệu là thật, và mô hình ngoại suy tốt ra khỏi vùng dữ liệu huấn luyện gần bằng nội suy — đúng chế độ hoạt động cần cho dự báo trước-khi-triển-khai.

---

## 13. RQ5 — Đồ thị nhân quả có giá trị; $do$-operator thì không (2:00)

**[SLIDE]**
- **Đồ thị phụ thuộc "đáng đồng tiền"**: thắng baseline uniform trên $85\%$ số cạnh in-distribution
- **$do$-operator không tạo lợi thế ước lượng** — lý do là *cấu trúc*, không phải thiết kế kém: can thiệp luôn áp tại **gateway (in-degree = 0)** → không có backdoor path để cắt → $P(Y\mid do(X)) = P(Y\mid X)$ *về mặt đồng nhất thức*
- 4 kiểm định độc lập xác nhận: không confounder đáng kể; backdoor adjustment không cải thiện transfer; counterfactual reasoning ≈ carry-residual

**Nói:** Đây là phần "trung thực trí tuệ" khác của bài: literature về causal RCA hay dùng $do$-operator để chẩn đoán — nhưng những can thiệp đó là *nội bộ* hệ thống (fault tiêm vào 1 service giữa đồ thị), có backdoor path thật sự. Bài toán của em thì khác: một tính năng khách hàng mới *luôn* vào từ biên hệ thống — gateway — nơi in-degree bằng 0 theo định nghĩa kiến trúc. Khi không có cạnh nào đi vào node bị can thiệp, việc "cắt cạnh" của $do$-operator không cắt được gì — interventional và observational distribution trùng nhau *identically*. Vì vậy máy móc causal phức tạp không mang lại lợi thế ở đây — nhưng đồ thị phụ thuộc (không phải $do$-calculus) vẫn cần thiết để lan truyền đúng đường.

---

## 14. Giới hạn & Threats to Validity (1:30)

**[SLIDE]**
- **Phạm vi**: 1 can thiệp/yêu cầu, tại gateway — phù hợp tính năng hướng khách hàng; **không hỗ trợ** background job, async consumer, thay đổi hành vi nội bộ không đổi tải ngoài
- **External validity**: 2 model, 2 topology — chưa đủ đại diện cho "quần thể generator"; frontier model chưa test
- **Gateway disambiguation** (nhiều gateway hợp lệ) chưa được kiểm — chỉ mới test *rejection* của gateway sai

**Nói:** Em muốn chủ động nêu giới hạn thay vì để phản biện phát hiện. Quan trọng nhất: framework chỉ xử lý can thiệp tại gateway — có lý do lý thuyết vững (slide 13), nhưng đồng nghĩa các tính năng không chạm biên ngoài (background job, worker bất đồng bộ) nằm ngoài phạm vi hiện tại.

---

## 15. Kết luận — thông điệp mang về (1:30)

**[SLIDE]**
> Khi một LLM sinh ra một object mà tính hợp lệ được định nghĩa bởi việc thuộc về một cấu trúc có thể kiểm tra được (schema, type system, call graph, kiến trúc hệ thống) — **cấu trúc đó kiểm tra được, và khi kiểm tra được, một chứng minh khả dụng mà không benchmark score nào thay thế được.**

- Verifier: **sound theo chứng minh + có tải trọng thực nghiệm** (load-bearing)
- Capability không thay thế verification (25× sai số ↓, GMR gần như không đổi)
- Proof cần được xác nhận lại khi triển khai (lỗi RQ3)
- Đồ thị nhân quả có giá trị; $do$-operator không — vì lý do cấu trúc của bài toán
- Đánh giá trung thực: kể cả khi phải rút lại kết quả cũ của chính mình

**Nói:** Tổng kết lại, câu hỏi ban đầu là: ta đảm bảo được gì về đầu ra của một LLM khi người tiêu thụ đầu ra đó là một pipeline tự động, không phải con người. Câu trả lời của bài báo không phải "dùng model tốt hơn" mà là: xây một lớp kiểm chứng tất định, chứng minh nó đúng, rồi *kiểm định thực nghiệm* rằng chứng minh đó có ý nghĩa trong thực tế — kể cả khi điều đó có nghĩa phải thừa nhận giới hạn hoặc rút lại kết quả cũ. Em xin dừng phần trình bày ở đây, sẵn sàng nhận câu hỏi.

---

## 16. Chuẩn bị Q&A — các câu hỏi khả năng cao sẽ gặp

Đây không phải slide — đọc trước để phản xạ nhanh khi được hỏi.

**Q: "Sound" nghĩa là gì ở đây, có phải là 'luôn đúng về mặt nghiệp vụ' không?**
→ Không. *Sound* ở đây chỉ đảm bảo tính hợp lệ **cấu trúc** (đúng service tồn tại, đúng gateway, đúng khoảng giá trị) — không đảm bảo con số dự báo là *chính xác*. Đó là lý do RQ1 tách riêng structural validity (GMR/SHR/PBVR) khỏi semantic accuracy (Anchor MAE).

**Q: Sao không dùng $do$-calculus đầy đủ nếu đã dùng Pearl's SCM?**
→ Chính vì đã phân tích kỹ bằng SCM nên biết *không cần* — can thiệp luôn ở node in-degree 0 nên $do(X)$ và $P(\cdot\mid X)$ trùng nhau identically (backdoor criterion). Dùng bộ máy $do$-calculus đầy đủ ở đây là "expensive no-op". Đây là kết quả RQ5, không phải thiếu sót.

**Q: Kết quả chỉ test trên 2 hệ thống demo (Sock Shop, Train Ticket) — liệu có tổng quát được không?**
→ Đã nêu ở phần giới hạn (slide 14): đây là external validity threat thật sự, thừa nhận công khai. Nhưng đóng góp cốt lõi (proof + phương pháp luận đánh giá) độc lập với 1 hệ thống cụ thể — RQ3 chính là thực nghiệm kiểm tra "liệu proof có transfer" khi đổi hệ thống, và nó *tìm ra lỗi* — đó là bằng chứng phương pháp hoạt động đúng như kỳ vọng.

**Q: Vì sao lại chọn rút lại (retract) một kết quả cũ, thay vì âm thầm sửa?**
→ Vì đó chính là luận điểm chính của RQ4: một protocol đánh giá tốt phải *có khả năng* trả lời "kết quả này không có ý nghĩa" — và khi protocol mới thực sự bắt được lỗi cũ, công bố công khai là bằng chứng protocol hoạt động, không phải là điểm yếu để giấu.

**Q: 25× giảm sai số ở RQ1 có ý nghĩa thống kê không, hay chỉ 1 lần chạy?**
→ 3 lần lặp độc lập (temperature 0.2) với Wilson 95% CI trên pooled calls (nêu ở phương pháp thống kê, Demšar/Holm-correction cho multiple comparison) — không phải một lần chạy may rủi.

---

*Gợi ý luyện tập: đọc to toàn bộ 1 lần với đồng hồ bấm giờ — nếu vượt quá ~24 phút, cắt bớt slide 8 (mô hình lan truyền tải) xuống còn 1 câu, vì nó không mang số liệu mới, chỉ là bối cảnh cho RQ4/RQ5.*
