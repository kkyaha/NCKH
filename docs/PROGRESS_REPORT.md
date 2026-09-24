# Báo cáo tiến độ nghiên cứu

> ⚠️ **TÀI LIỆU THEO HỆ ĐÁNH SỐ RQ CŨ (RQ1–RQ12).** Số RQ trong file này **không khớp**
> khung 5 RQ hiện tại của `docs/paper_draft.tex`. Xem bảng dịch ở đầu `README.md` gốc.
> Giữ lại làm hồ sơ gốc; **không dùng làm nguồn số liệu**.

**Ngày:** 16/09/2026
**Đề tài:** Xác minh và chứng nhận đầu ra LLM trong pipeline phân tích dung năng microservice (dự phóng tải cho yêu cầu tính năng chưa xây)

---

## 1. Bối cảnh và bộ khung ban đầu

Hệ thống gồm bốn thành phần: Parser Agent (dịch yêu cầu ngôn ngữ tự nhiên thành can thiệp $do(X{=}\delta)$ có kiểm chứng ba lớp), Architecture Agent (lan truyền blast radius), Capacity Agent (mô hình SCM hai tầng dự phóng CPU/Memory/Latency), và một LLM tổng hợp báo cáo. Bản thảo ban đầu đóng khung quanh luận điểm *"Capability Is Not Verification"* với 5 câu hỏi nghiên cứu (RQ1–RQ5) và hai mệnh đề trung tâm: Proposition 1 (bộ xác minh $\Pi$ là một phép chiếu toàn phần, tất định, bảo đảm đúng cho mọi đầu ra LLM) và một chuỗi thực nghiệm hỗ trợ (backend sweep, Scope Gate conformal, port sang Train Ticket, benchmark RCAEval/Alibaba, do-operator vs conditioning).

## 2. Vấn đề phát hiện khi rà soát lại

Khi đánh giá lại nghiêm túc bộ 5 RQ, ba nhóm vấn đề nổi lên:

**(a) Claim trung tâm (RQ1) mỏng về mẫu.** Luận điểm "capability không thay được verification" chỉ dựa trên $n{=}2$ backend, backend thứ hai chỉ có 1 repeat (giới hạn quota). Bộ 50 prompt đánh giá do một tác giả tự viết, chưa có kiểm định độc lập.

**(b) Corollary 1 khiến bảng kết quả trung tâm mang tính tautology.** PBVR và SHR của cấu hình có guard bằng $0$ theo định nghĩa (do $\Pi$ là phép chiếu toàn phần) — nghĩa là hai trong ba cột của bảng RQ1 không đo được gì mới, toàn bộ giá trị thực nghiệm phải dồn vào các tập đối kháng nhỏ.

**(c) Không có dataset công khai nào vừa đủ tên dịch vụ thật vừa đủ biến động tải.** RCAEval (Sock Shop, Train Ticket) có tên dịch vụ thật nhưng tải được giữ ổn định do thiết kế cho bài toán chẩn đoán lỗi; Alibaba có tải biến động thật nhưng tên dịch vụ bị hash. Dùng RCAEval để đánh giá dự báo RAM (MAPE $2$–$4\%$) từng cho kết quả trông tốt nhưng vô nghĩa: $R^2=0.0034$ cho thấy workload gần như không giải thích được phương sai của RAM trên dataset này, nên sai số thấp chỉ phản ánh RAM gần như bất biến chứ không phải mô hình học được quan hệ nào.

Từ đó, quyết định xây dựng lại toàn bộ khung câu hỏi nghiên cứu và tìm một trục kỹ thuật mới đủ sức nặng, thay vì chỉ chỉnh sửa câu chữ.

## 3. Hướng đã đi và kết quả cụ thể

### 3.1. Tái cấu trúc quanh nguyên lý "Admissible Evaluation"

Đề xuất một định nghĩa thống nhất: một phép đánh giá $E$ là *khả bác* (admissible) đối với một mệnh đề $C$ nếu tồn tại kết cục của $E$ bác bỏ được $C$; ngược lại là *bất khả bác* (vacuous). Từ đó, 5 RQ được viết lại thành 5 cặp (mệnh đề / phép đánh giá thay thế có khả bác), thay vì 5 chủ đề rời rạc. Viết lại Abstract, Introduction, Conclusion, và bảng tổng hợp `tab:rq-spine` theo khung này.

**Đã kiểm tra tính mới qua tra cứu thực tế** (không dựa vào suy luận): tìm thấy các công trình rất gần (2025–2026) — CALLMIT (LLM+causal graph cho performance testing), PASC (conformal prediction đa tầng cho LLM pipeline), Kotte 2026 (impossibility result cho conformal risk control — tổng quát hoá đúng phát hiện thực nghiệm ở RQ2), "Proof-Carrying Certificates for LLM Pipelines" (Lean 4, cùng triết lý với Proposition 1/2 nhưng nghiêm ngặt hơn). Định vị lại đóng góp cho chính xác, tránh tuyên bố "chưa ai làm" mà không kiểm chứng.

### 3.2. Đào sâu Proposition 2 — Certified Capacity Envelope

Chứng minh: vì mọi mechanism trong hai tầng SCM đều đơn điệu không giảm ($\beta\ge0$ do ràng buộc NNLS, và $\phi(w)=w/(c-w)$ đơn điệu tăng), toàn bộ phép lan truyền là hợp thành đơn điệu theo $\delta$. Do đó **2 lượt forward tất định** (tại $\delta{=}5\%$ và $\delta{=}50\%$, hai đầu khoảng đã được Proposition 1 bảo đảm) chứng nhận một khoảng chứa giá trị điểm-ước-lượng của mọi node hạ nguồn, với mọi $\delta^*$ mà Layer 3 có thể phát ra — không tốn thêm lệnh gọi LLM nào.

**Cài đặt và kiểm tra trên code thật** (`src/agents/capacity_agent.py`), không chỉ trên giấy:

- Viết `_deterministic_forward`, `_check_monotone_precondition`, `certified_envelope` — kiểm tiền đề $\beta\ge0$ trên mô hình đã fit thật, không tin ràng buộc solver.
- **Phát hiện một defect thật**: cả $7/7$ mechanism latency của Sock Shop và $28/28$ của Train Ticket vi phạm $\beta\ge0$ — do `QueueingLatencyRegressor` dùng hồi quy không ràng buộc trên hai đặc trưng cộng tuyến ($W$ thô và $\phi(W)$). Có phản ví dụ cụ thể: `orders_latency-50` dự báo latency ở $\delta{=}50\%$ **thấp hơn** ở $\delta{=}5\%$.
- **Sửa bằng đúng ràng buộc đã dùng cho CPU/Mem** (`positive=True`). Kết quả không phải đánh đổi: MAPE held-out cải thiện $7/7$ dịch vụ (Sock Shop, trung vị $33.9\%\to12.1\%$) và $23/28$ dịch vụ (Train Ticket, có báo cáo trung thực $5$ ca tệ đi $+1.8$ đến $+18.6$ điểm phần trăm).
- Cơ sở lý thuyết: tra cứu và trích dẫn Slawski & Hein (2013) — NNLS có tính tự-điều-chuẩn (implicit regularization), giải thích vì sao ràng buộc "miễn phí" ở phần lớn trường hợp.

### 3.3. Mở rộng Proposition 3 — Joint Intervention Envelope

Câu hỏi mới: khi hai yêu cầu được duyệt cùng lúc, chứng chỉ có cộng được đơn giản không? Chứng minh hai phần: (a) tầng tuyến tính (CPU/Mem/Workload) cộng **đúng tuyệt đối** (superposition); (b) tầng latency (lồi) **siêu cộng tính** — hiệu ứng thật của hai can thiệp hội tụ tại cùng một điểm nghẽn hàng đợi luôn $\ge$ tổng hai hiệu ứng tính riêng lẻ (bất đẳng thức Jensen). Hệ quả thực tiễn: kiểm từng yêu cầu an toàn riêng lẻ là **cần nhưng không đủ**.

- Kiểm trên collider tổng hợp (dùng đúng lớp mechanism sản xuất): gap tuyến tính $=0.000000$ tuyệt đối; gap latency dương và tăng đơn điệu theo $\delta$ ($+0.028\to+4.774$).
- **Kiểm trên Alibaba thật, rồi rút lại vì không bền vững**: quét toàn bộ 13 node có fan-in$\ge2$ trong call graph thật. $10/13$ thiếu dữ liệu workload+response-time đồng thời; 2 ca thật còn lại ban đầu cho gap dương ($+5.09\%$, $+1.11\%$). Nhưng khi mở rộng cửa sổ dữ liệu từ 5h lên 10h để kiểm tính ổn định, **cả hai gap sụp về đúng $0$**. Nguyên nhân: `capacity_` (trần năng lực) được đặt bằng $1.5\times$ giá trị workload lớn nhất **từng quan sát được** — khi cửa sổ rộng hơn lộ ra một giá trị cao hơn, trần bị đẩy lên, làm toàn bộ dữ liệu cũ trông "còn xa trần" hơn, khiến NNLS coi thành phần phi tuyến (queueing) không còn cần thiết và ép hệ số về $0$. Đây là một lỗ hổng thật trong cách ước lượng trần năng lực (nhạy với cỡ mẫu), không phải Proposition 3/4 sai — nhưng **đã rút lại số liệu thật khỏi bài**, chỉ giữ bằng chứng trên dữ liệu tổng hợp (vẫn đúng, vì trần ở đó được đặt cố định, không phụ thuộc quan sát).
- Đã báo cáo trung thực giới hạn: $n{=}2$ là minh chứng hiện tượng tồn tại, chưa phải phân phối thống kê.

### 3.4. Quyết định về phạm vi

Đã cân nhắc phương án đưa Proposition 3 thành trục duy nhất của toàn bài (pivot hoàn toàn), nhưng sau khi kiểm chứng thực nghiệm ($n{=}2$ trên dữ liệu thật), quyết định **không pivot toàn bộ** — giữ nguyên bề rộng đã có (Alibaba $n{=}1\,293$ cho RQ4, RQ5a $n{=}27$ cạnh, do-operator 4 kiểm định độc lập, RQ2 conformal) và đưa Proposition 2+3 vào làm nội dung chính của RQ1 (đổi tên "Necessity & Composability"), thay vì tách RQ6 riêng hay xoá bỏ phần còn lại.

## 4. Khó khăn / vấn đề đang gặp phải

1. **Backend sweep chưa chạy.** Đây vẫn là điểm yếu lớn nhất của claim trung tâm RQ1 ($n{=}2$). Đã xác nhận hạ tầng sẵn sàng (9 backend, credentials đủ 3 vendor) và chốt phạm vi (2 backend tuyến đầu — `gemini-3.1-pro`, `gpt-4o` — 1 repeat mỗi backend), nhưng **chưa thực thi** vì cần cân nhắc chi phí API thật trước khi bắn hàng loạt lệnh gọi.
2. **Proposition 3/4 hiện không có bằng chứng thật nào đứng vững** — ca thật duy nhất tìm được đã bị rút lại (mục 3.3). Chỉ còn bằng chứng trên dữ liệu tổng hợp.
3. **`QueueingLatencyRegressor` có một lỗ hổng thật, độc lập với lỗi đã sửa ở mục 3.2**: `capacity_` ước lượng bằng $1.5\times\max$ quan sát được, không ổn định theo cỡ mẫu. Chưa sửa — mới dừng ở mức phát hiện và rút lại kết luận bị ảnh hưởng.
4. **`RQ_FRAMEWORK_V3.md`** (tài liệu thiết kế nội bộ) chưa đồng bộ với cấu trúc RQ1 mới nhất.

## 5. Hướng tiếp theo đề xuất

Theo thứ tự ưu tiên:

1. **Quyết định và chạy backend sweep** — cần chốt ngân sách/thời gian trước khi thực thi (~vài trăm lệnh gọi live LLM, 2 backend tuyến đầu, 1 repeat mỗi backend). Đây là việc duy nhất còn lại có thể vá được điểm yếu $n$ của claim trung tâm.
2. **Sửa lại cách ước lượng `capacity_`** (dùng phân vị ổn định, ví dụ P99 của tập huấn luyện, thay vì $1.5\times\max$) rồi kiểm lại xem hiện tượng siêu cộng tính có tái xuất hiện trên dữ liệu thật không.
3. Đồng bộ `RQ_FRAMEWORK_V3.md`.
4. (Tuỳ chọn, chi phí cao hơn) Hình thức hoá Proposition 1–4 bằng Lean/Coq theo đúng chuẩn mực bài "Proof-Carrying Certificates" (2026) đã tra cứu được, nếu muốn nâng độ nghiêm ngặt lên mức machine-checked.

---

*Tài liệu này tổng hợp từ quá trình làm việc thực tế trên codebase (`src/agents/capacity_agent.py`) và bản thảo (`docs/paper_draft.tex`); mọi số liệu nêu trên đều đã được kiểm chứng bằng code chạy thật, không phải ước tính.*
