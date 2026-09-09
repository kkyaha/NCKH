# 📊 BÁO CÁO RQ5 — CHI PHÍ ĐIỀU PHỐI ĐA TÁC TỬ vs. SINGLE-LLM-CALL

Tài liệu này được **sinh tự động** từ `rq5_coordination_overhead.csv` để trả lời RQ5:
> **RQ5**: *"Kiến trúc đa tác tử (Parser Agent + Architecture Agent + Capacity Agent
> với dual-path SCM) có vượt trội một lời gọi LLM đơn lẻ (single-LLM-call) không,
> và chi phí điều phối đa tác tử (số lần gọi LLM, độ trễ) đem lại giá trị gì?"*

> ⚠️ **CẢNH BÁO**: Chạy với bộ giả lập offline (`SYNTHETIC_OFFLINE_EMULATOR_DO_NOT_CITE_AS_LLM_RESULT`), KHÔNG phải LLM thật. Không dùng số liệu này làm bằng chứng khoa học.

Thời điểm chạy: `2026-09-09T09:32:40.759072` | Số lần lặp: `1` | Backend: `SYNTHETIC_OFFLINE_EMULATOR_DO_NOT_CITE_AS_LLM_RESULT`

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
| `Single_LLM_Call` | 8.00% | 12.00% | 10.00% | 29.40% | 1.00 | 0.03 ms |
| `Guarded_MAS_Pipeline` | 0.00% | 0.00% | 0.00% | 2.80% | 1.52 | 0.08 ms |

*Ghi chú*: `Single_LLM_Call` phải tự đoán CẢ phần capacity assessment trong CÙNG một
lần gọi — đây là cấu hình khó hơn B1/B2 của RQ3 (vốn chỉ phải làm mỗi việc parse).

---

## 🤝 2. STATUS AGREEMENT (Single_LLM_Call so với phán quyết SCM $do(x)$ thật)

Tổng thể: **2.33%** (n=43 prompt trong phạm vi taxonomy,
loại các prompt bị G6 từ chối ở nhánh Guarded_MAS vì không có status để đối chiếu).

Theo nhóm:

| Category | Status Agreement (%) |
|---|---|
| Adversarial_Stress | 0.0 |
| Complex_MultiHop | 0.0 |
| In_Distribution | 5.0 |
| Subtle_ReadOnly | 0.0 |


---

## 📁 3. DỮ LIỆU THÔ
* Toàn bộ 100 bản ghi (50 prompts × 2 cấu hình × 1 lần lặp):
  `data/processed/scm_results/rq5_coordination_overhead.csv`
* Chi tiết status agreement theo từng prompt: `data/processed/scm_results/rq5_status_agreement.csv`
