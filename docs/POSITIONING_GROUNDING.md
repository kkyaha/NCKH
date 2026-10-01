# Định vị: ba lớp vi phạm, và lớp mà năng lực mô hình không giải quyết

> Mục này thay cho cách dùng từ `structural` hiện tại trong `paper_draft.tex`. Lý do đổi ở §3.
> Số liệu: `data/processed/scm_results/parser_ablation_benchmark*.csv` (600 dòng live Gemini,
> 200 dòng live gpt-oss-120b — đã xác minh cả hai mang nhãn `LIVE_`, `SYNTHETIC = 0`).

## 1. Ba lớp vi phạm, không phải một

Gộp chúng thành "structural" làm mất đúng phát hiện của chúng ta: **ba lớp phản ứng khác
nhau với năng lực mô hình.**

| lớp | định nghĩa | chỉ số | hỏi gì |
|---|---|---|---|
| **neo thực tế** (grounding) | thực thể được nêu có **tồn tại trong hệ đang chạy** không; điểm tiêm có phải gateway thật không | `SHR` = service không có trong topology<br>`GMR` = node tiêm không phải node in-degree 0 | *mô hình có biết hệ này không?* |
| **biên hợp lý** (plausibility) | giá trị có nằm trong khoảng vật lý cho phép | `PBVR` = `Δ ∉ [5%, 50%]` | *con số có khả dĩ không?* |
| **độ lớn** (magnitude) | sai số so với anchor hiệu chỉnh | `MAE` | *con số có đúng không?* |

## 2. Năng lực mô hình giải quyết hai lớp, không giải quyết lớp thứ ba

Cùng 50 prompt, cùng harness, chỉ đổi backend:

| | `gemini-flash-lite` (3 lần lặp) | `gpt-oss-120b` (117B, **1 lần lặp**) | năng lực giúp? |
|---|---|---|---|
| **độ lớn** `MAE` | 844,6 ± 568,2 % | **33,6 %** | ✅ **25×** |
| **biên hợp lý** `PBVR` | 38,0 ± 2,0 % | **18,0 %** | ✅ **giảm nửa** |
| **neo thực tế** `SHR` | 91,3 ± 1,2 % | **86,0 %** | ❌ −5,3 điểm |
| **neo thực tế** `GMR` | 100,0 ± 0,0 % | **92,0 %** | ❌ −8,0 điểm |

`GMR` của mô hình 117B theo **từng nhóm prompt**: In-Distribution **90%**, Complex Multi-Hop
**100%**, Subtle Read-Only **100%**, Adversarial Stress **80%** — nên đây là thất bại **hệ
thống**, không phải phản ứng với độ khó của prompt.

## 3. Vì sao phải đổi tên `structural` → `grounding`

`arXiv 2607.26220` (*Model-Driven Requirements Configuration with Three-Valued Uncertainty
Scoring*, 2026) dùng **cùng từ `structural`** cho **một lớp khác**: ràng buộc nội bộ của một
lattice cấu hình — thiếu node con bắt buộc, chọn sai số lượng nhánh loại trừ, node mồ côi.

Và kết luận của họ **ngược chiều** với ta:

| | họ (2607.26220) | ta |
|---|---|---|
| lớp vi phạm | nhất quán nội bộ của lattice | **neo vào hệ triển khai** |
| Llama 3.1 8B | 0,39% còn lại (6/1535 quyết định) | — |
| **mô hình frontier** | **0% trên 37/37 vision** | **86–92% ở mô hình 117B** |
| kết luận | *mô hình mạnh hơn xoá sạch vi phạm* | *mô hình mạnh hơn **không** xoá được* |

Giữ từ `structural` cho lớp của ta thì người đọc sẽ hiểu là ta mâu thuẫn với một bài đã xuất
bản. Thực tế **không mâu thuẫn** — hai lớp khác nhau, và chúng khác nhau ở đúng một điểm:

> Điền một schema cho đúng là bài toán **tuân thủ chỉ dẫn** — năng lực giải quyết được.
> Biết `carts` có tồn tại trong hệ này và request phải vào qua `front-end` là **tri thức về
> một hệ thống cụ thể** — năng lực không cung cấp được, vì nó không nằm trong mô hình.

## 4. Câu định vị dùng cho bài

> Năng lực mô hình giải quyết được ràng buộc **nhất quán nội bộ** của đặc tả — Sharma et al.
> báo 0% vi phạm với một mô hình frontier trên 37/37 cấu hình — và trong thiết lập của chúng
> tôi nó cải thiện **độ lớn** 25× và **biên hợp lý** một nửa. Nhưng nó **không** giải quyết
> được vi phạm **neo thực tế**: chuyển sang mô hình 117B của một nhà cung cấp khác chỉ hạ tỉ
> lệ chỉ sai gateway từ 100,0% xuống 92,0% và tỉ lệ bịa service từ 91,3% xuống 86,0%, đồng
> đều trên cả bốn nhóm prompt. Hai lớp này đòi hai loại verifier khác nhau, và loại thứ hai
> **không suy được từ đặc tả** — nó cần trạng thái của hệ đang triển khai.

## 5. So với validator của họ, `Π` mạnh hơn ở hai điểm

| | 2607.26220 | 2607.03651 (capacity, miền vận tải) | ta |
|---|---|---|---|
| verifier tất định | ✅ hàm Python | ❌ chỉ vòng phản hồi heuristic | ✅ |
| **chứng minh cho MỌI đầu ra** | ❌ chỉ thực nghiệm, ngân sách 150 lần gọi | ❌ | ✅ `Π` toàn phần |
| **từ chối khi ngoài phạm vi** | ❌ | ❌ ("LLM được chỉ thị ra quyết định bất kể") | ✅ Scope Gate |
| ground truth | cấu hình hợp lệ theo lattice | **tổng hợp** (tác giả tự tiêm hệ số nhân) | **đo được** từ 18 tính năng cài thật |
| miền hiệu năng/SLO | ❌ nói rõ không chạm | ✅ nhưng là hub vận tải | ✅ |

## 6. Hai việc phải làm trước khi dùng mục này

1. **Chạy lại nhánh 117B ×3 lần lặp.** Xác minh từ dữ liệu: `parser_ablation_benchmark.csv`
   có `repeat_id ∈ {1,2,3}` (150 dòng/cấu hình); `..._gpt-oss-120b.csv` chỉ có `repeat_id = 1`
   (50 dòng/cấu hình). So sánh tiêu đề đang là **3 lần lặp với 1 lần lặp** và sẽ bị bắt.
2. **Không đồng nhất `PBVR` với `structural` của họ.** `Δ ∉ [5%,50%]` là **biên hợp lý của
   miền**, không phải ràng buộc schema. Trình bày ba lớp như §1 và nói rõ lớp của họ là lớp
   **thứ tư** — như vậy xung đột thuật ngữ mất hẳn thay vì được che.

## 7. Bảng LaTeX

```latex
\begin{table}[t]
\centering
\caption{Ba lớp vi phạm phản ứng khác nhau với năng lực mô hình. Cùng 50 prompt, cùng
harness, chỉ đổi backend. Lớp \emph{neo thực tế} hầu như không đổi.}
\label{tab:three-classes}
\begin{tabular}{llccc}
\toprule
Lớp & Chỉ số & \texttt{gemini-flash-lite} & \texttt{gpt-oss-120b} & Thay đổi \\
\midrule
Độ lớn        & MAE  & $844.6\pm568.2\%$ & $33.6\%$ & $25\times$ tốt hơn \\
Biên hợp lý   & PBVR & $38.0\pm2.0\%$    & $18.0\%$ & giảm nửa \\
\midrule
Neo thực tế   & SHR  & $91.3\pm1.2\%$    & $86.0\%$ & $-5.3$ đ \\
Neo thực tế   & GMR  & $100.0\pm0.0\%$   & $92.0\%$ & $-8.0$ đ \\
\bottomrule
\end{tabular}
\end{table}
```
