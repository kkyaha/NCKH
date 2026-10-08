# 📊 BÁO CÁO BENCHMARK VÀ THỰC NGHIỆM ABLATION CHO LLM PARSER (RQ3)
## Đề Tài: *Grounded Large Language Models for Structural Causal Intervention in Microservices*

Tài liệu này được **sinh tự động** từ `parser_ablation_benchmark.csv` (không viết tay số liệu)
để giải quyết **Research Question 3 (RQ3)**:
> **RQ3:** *"Mô hình Grounded LLM Parser chuyển dịch yêu cầu ngôn ngữ tự nhiên thành đại số can thiệp $do(x)$ với độ tin cậy ra sao, và các Runtime Guards triệt tiêu hiện tượng ảo giác (Hallucination) và vi phạm biên vật lý như thế nào?"*

> ✅ Backend: **LLM thật** (`LIVE_openai/gpt-oss-120b`), 3 lần lặp độc lập (nhiệt độ 0.2, không deterministic) để đo phương sai run-to-run.

Thời điểm chạy: `2026-10-08T11:25:25.870900` | Số lần lặp: `3` | Backend: `LIVE_openai/gpt-oss-120b`

---

## 🏛️ 1. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Quy mô tập dữ liệu**: **50 kịch bản yêu cầu tính năng phần mềm e-commerce** (`data/benchmark/parser_benchmark_prompts.json`), phân bổ vào 4 nhóm: In-Distribution (20), Complex Multi-Hop (10), Subtle Read-Only (10), Adversarial/Stress-Test (10).
* **4 Cấu hình đối chiếu**: `Unguarded_ZeroShot_LLM`, `Unguarded_FewShot_LLM` (LLM thật, không guard), `Rule_Only` (heuristic từ khóa, không LLM), `Guarded_Hybrid_Parser` (đề xuất — fast-path bảng hiệu chỉnh khi similarity ≥ 0.6, LLM có guard khi không match rõ).

### ⚠️ Giới hạn phương pháp luận cần lưu ý khi đọc bảng dưới đây
1. **`expected_anchor`** (dùng để tính MAE) là **cùng bộ giá trị hiệu chỉnh cứng** (`CALL_CHAINS` trong `request_router.py`) mà `Rule_Only` và nhánh fast-path của `Guarded_Hybrid_Parser` tra cứu trực tiếp. Vì vậy MAE thấp của 2 cấu hình này ở nhóm **In-Distribution** phần lớn phản ánh việc tra bảng đúng, **không phải** năng lực suy luận ngữ nghĩa độc lập — MAE có ý nghĩa kiểm chứng thật sự nằm ở nhóm **Complex_MultiHop / Subtle_ReadOnly / Adversarial_Stress**, nơi cấu hình phải suy luận vượt ra ngoài match từ khóa trực tiếp.
2. Các cấu hình `Unguarded_*` gọi LLM thật không có bảng hiệu chỉnh trong ngữ cảnh (ZeroShot) hoặc có (FewShot) — đây là phép so sánh công bằng về khả năng tự kiềm chế của LLM khi không có runtime guard, không phải baseline bị lập trình để thất bại.
3. LLM thật có tính ngẫu nhiên (temperature=0.2) — bảng dưới báo cáo **mean ± std qua 3 lần lặp độc lập**, không phải một lần chạy duy nhất.

---

## 📊 2. BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHIẾU (RQ3 MASTER TABLE)

| Cấu Hình Mô Hình | PBVR (%) | SHR (%) | GMR (%) | Anchor MAE (%) | Adv. PBVR (%) | Latency (ms) | Fast-Path Bypass (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Unguarded_ZeroShot_LLM`  | 9.33±8.33% | 61.33±49.65% | 66.00±53.70% | 29.33±20.27% | 36.67±35.12% | 1327.32±79.41 ms | 0.00±0.00% |
| `Unguarded_FewShot_LLM`  | 12.67±15.53% | 6.67±7.02% | 48.00±43.31% | 10.44±4.56% | 10.00±17.32% | 1537.86±455.60 ms | 0.00±0.00% |
| `Rule_Only`  | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 2.80±0.00% | 0.00±0.00% | 0.39±0.02 ms | 100.00±0.00% |
| `Guarded_Hybrid_Parser` 🏆 **(Đề xuất)** | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 2.65±0.25% | 0.00±0.00% | 4850.51±7874.99 ms | 0.00±0.00% |

*(Không suy diễn "0.0%" hay "vượt trội" nếu bảng trên không thực sự cho ra số đó — mọi tuyên bố kết luận phải đọc trực tiếp từ bảng số ở trên sau khi chạy.)*

---

## 📋 3. CHI TIẾT THEO TỪNG NHÓM PROMPT (BREAKDOWN BY CATEGORY)
Đây là bảng quan trọng nhất để đánh giá khả năng khái quát hoá thật sự (ngoài phạm vi tra bảng In-Distribution):

| Model | Category | N | PBVR (%) | SHR (%) | GMR (%) | MAE (%) |
|---|---|---|---|---|---|---|
| `Unguarded_ZeroShot_LLM` | Adversarial_Stress | 10 | 36.67 | 33.33 | 56.67 | 109.55 |
| `Unguarded_ZeroShot_LLM` | Complex_MultiHop | 10 | 6.67 | 70.0 | 70.0 | 15.22 |
| `Unguarded_ZeroShot_LLM` | In_Distribution | 20 | 0.0 | 68.33 | 68.33 | 7.64 |
| `Unguarded_ZeroShot_LLM` | Subtle_ReadOnly | 10 | 3.33 | 66.67 | 66.67 | 6.58 |
| `Unguarded_FewShot_LLM` | Adversarial_Stress | 10 | 10.0 | 0.0 | 23.33 | 17.55 |
| `Unguarded_FewShot_LLM` | Complex_MultiHop | 10 | 13.33 | 13.33 | 60.0 | 11.0 |
| `Unguarded_FewShot_LLM` | In_Distribution | 20 | 10.0 | 6.67 | 65.0 | 7.61 |
| `Unguarded_FewShot_LLM` | Subtle_ReadOnly | 10 | 20.0 | 6.67 | 26.67 | 8.4 |
| `Rule_Only` | Adversarial_Stress | 10 | 0.0 | 0.0 | 0.0 | 2.5 |
| `Rule_Only` | Complex_MultiHop | 10 | 0.0 | 0.0 | 0.0 | 4.5 |
| `Rule_Only` | In_Distribution | 20 | 0.0 | 0.0 | 0.0 | 2.5 |
| `Rule_Only` | Subtle_ReadOnly | 10 | 0.0 | 0.0 | 0.0 | 2.0 |
| `Guarded_Hybrid_Parser` | Adversarial_Stress | 10 | 0.0 | 0.0 | 0.0 | 1.67 |
| `Guarded_Hybrid_Parser` | Complex_MultiHop | 10 | 0.0 | 0.0 | 0.0 | 4.43 |
| `Guarded_Hybrid_Parser` | In_Distribution | 20 | 0.0 | 0.0 | 0.0 | 2.58 |
| `Guarded_Hybrid_Parser` | Subtle_ReadOnly | 10 | 0.0 | 0.0 | 0.0 | 2.0 |

---

## 📄 4. BẢNG MÃ NGUỒN LATEX CHO BÀI BÁO
Đã được xuất tự động tại: [`docs/table_rq3_parser_ablation.tex`](../docs/table_rq3_parser_ablation.tex).

## 📁 5. DỮ LIỆU THÔ
* Toàn bộ 600 bản ghi (50 prompts × 4 cấu hình × 3 lần lặp): `data/processed/scm_results/parser_ablation_benchmark.csv`
* Breakdown theo category: `data/processed/scm_results/parser_ablation_by_category.csv`
