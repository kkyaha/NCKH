# 📊 BÁO CÁO RQ5 — CHI PHÍ ĐIỀU PHỐI ĐA TÁC TỬ vs. SINGLE-LLM-CALL

Tài liệu này được **sinh tự động** từ `rq5_coordination_overhead.csv` để trả lời RQ5:
> **RQ5**: *"Kiến trúc đa tác tử (Parser Agent + Architecture Agent + Capacity Agent
> với dual-path SCM) có vượt trội một lời gọi LLM đơn lẻ (single-LLM-call) không,
> và chi phí điều phối đa tác tử (số lần gọi LLM, độ trễ) đem lại giá trị gì?"*

> ✅ Backend: **LLM thật** (`LIVE_gemini-flash-lite-latest`), 3 lần lặp độc lập.

Thời điểm chạy: `2026-09-09T17:55:13.264042` | Số lần lặp: `3` | Backend: `LIVE_gemini-flash-lite-latest`

---

## ⚠️ Giới hạn diễn giải cần đọc trước bảng số liệu

Không tồn tại "ground truth" thực tế cho một tính năng **chưa được xây dựng** — vì vậy
"Status Agreement" dưới đây **không phải** độ chính xác so với sự thật, mà là tỔc độ
`Single_LLM_Call` trùng khớp với phán quyết mà mô phỏng SCM $do(x)$ thật sự đưa ra
(`Guarded_MAS_Pipeline`, được tính từ các mô hình nhân quả đã huấn luyện trên dữ liệu
RCAEval thật, không phải LLM đoán). Bất đồng không tự động nghĩa là `Single_LLM_Call`
sai — nhưng nó **không có cơ chế nào để tự kiểm chứng lại chính nó** (không SCM, không
guard), nên không có cách nào phân biệt được hai trường hợp đó chỉ từ đầu ra của nó.
Đây chính là luận điểm RQ5 muốn đo: chi phí điều phối đa tác tử đổi lấy khả năng
**giải thích và kiểm chứng được**, không chỉ là một con số cuối cùng.

---

## 📊 1. BẢNG TỔNG HỢP (PBVR/SHR/GMR/MAE giống hệt phương pháp RQ3, cộng thêm chi phí)

| Configuration | PBVR (%) | SHR (%) | GMR (%) | Anchor MAE (%) | LLM calls/prompt | LLM latency (ms)/prompt |
|---|---|---|---|---|---|---|
| `Single_LLM_Call` | 27.33±1.15% | 1.33±1.15% | 2.00±0.00% | 53.55±1.01% | 1.00±0.00 | 1683.62±603.12 ms |
| `Guarded_MAS_Pipeline` | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 2.27±0.06% | 1.52±0.00 | 4053.55±1190.01 ms |

*Ghi chú*: `Single_LLM_Call` phải tự đoán CẢ phần capacity assessment trong CÙNG một
lần gọi — đây là cấu hình khó hơn B1/B2 của RQ3 (vốn chỉ phải làm mỗi việc parse).

---

## 🤝 2. STATUS AGREEMENT (Single_LLM_Call so với phán quyết SCM $do(x)$ thật)

Tổng thể: **46.51%** (n=129 prompt trong phạm vi taxonomy,
loại các prompt bị G6 từ chối ở nhánh Guarded_MAS vì không có status để đối chiếu).

Theo nhóm:

| Category | Status Agreement (%) |
|---|---|
| Adversarial_Stress | 22.22 |
| Complex_MultiHop | 70.0 |
| In_Distribution | 41.67 |
| Subtle_ReadOnly | 47.62 |


---

## 📁 3. DỮ LIỆU THÔ
* Toàn bộ 300 bản ghi (50 prompts × 2 cấu hình × 3 lần lặp):
  `data/processed/scm_results/rq5_coordination_overhead.csv`
* Chi tiết status agreement theo từng prompt: `data/processed/scm_results/rq5_status_agreement.csv`
