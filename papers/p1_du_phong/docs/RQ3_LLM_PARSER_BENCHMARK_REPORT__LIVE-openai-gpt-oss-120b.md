# 📊 BÁO CÁO BENCHMARK VÀ THỰC NGHIỆM ABLATION CHO LLM PARSER (RQ3)
## Đề Tài: *Grounded Large Language Models for Structural Causal Intervention in Microservices*

Tài liệu này được **sinh tự động** từ `parser_ablation_benchmark.csv` (không viết tay số liệu)
để giải quyết **Research Question 3 (RQ3)**:
> **RQ3:** *"Mô hình Grounded LLM Parser chuyển dịch yêu cầu ngôn ngữ tự nhiên thành đại số can thiệp $do(x)$ với độ tin cậy ra sao, và các Runtime Guards triệt tiêu hiện tượng ảo giác (Hallucination) và vi phạm biên vật lý như thế nào?"*

> ✅ Backend: **LLM thật** (`LIVE_openai/gpt-oss-120b`), 3 lần lặp độc lập (nhiệt độ 0.2, không deterministic) để đo phương sai run-to-run.

Thời điểm chạy: `2026-10-11T23:57:44.345480` | Số lần lặp: `3` | Backend: `LIVE_openai/gpt-oss-120b`

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
| `Unguarded_ZeroShot_LLM`  | 12.67±3.06% | 92.00±0.00% | 95.33±1.15% | 34.55±1.49% | 46.67±5.77% | 1151.96±115.96 ms | 0.00±0.00% |
| `Unguarded_FewShot_LLM`  | 34.00±7.21% | 10.00±5.29% | 78.67±5.03% | 16.43±1.45% | 40.00±0.00% | 1414.42±106.40 ms | 0.00±0.00% |
| `Rule_Only`  | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 2.80±0.00% | 0.00±0.00% | 0.38±0.03 ms | 100.00±0.00% |
| `Guarded_Hybrid_Parser` 🏆 **(Đề xuất)** | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 2.40±0.08% | 0.00±0.00% | 18308.80±6852.94 ms | 0.00±0.00% |

*(Không suy diễn "0.0%" hay "vượt trội" nếu bảng trên không thực sự cho ra số đó — mọi tuyên bố kết luận phải đọc trực tiếp từ bảng số ở trên sau khi chạy.)*

---

## 📋 3. CHI TIẾT THEO TỪNG NHÓM PROMPT (BREAKDOWN BY CATEGORY)
Đây là bảng quan trọng nhất để đánh giá khả năng khái quát hoá thật sự (ngoài phạm vi tra bảng In-Distribution):

| Model | Category | N | PBVR (%) | SHR (%) | GMR (%) | MAE (%) |
|---|---|---|---|---|---|---|
| `Unguarded_ZeroShot_LLM` | Adversarial_Stress | 10 | 46.67 | 60.0 | 76.67 | 128.0 |
| `Unguarded_ZeroShot_LLM` | Complex_MultiHop | 10 | 10.0 | 100.0 | 100.0 | 25.58 |
| `Unguarded_ZeroShot_LLM` | In_Distribution | 20 | 1.67 | 100.0 | 100.0 | 7.26 |
| `Unguarded_ZeroShot_LLM` | Subtle_ReadOnly | 10 | 3.33 | 100.0 | 100.0 | 4.65 |
| `Unguarded_FewShot_LLM` | Adversarial_Stress | 10 | 40.0 | 6.67 | 63.33 | 46.0 |
| `Unguarded_FewShot_LLM` | Complex_MultiHop | 10 | 40.0 | 16.67 | 96.67 | 17.44 |
| `Unguarded_FewShot_LLM` | In_Distribution | 20 | 15.0 | 5.0 | 95.0 | 5.69 |
| `Unguarded_FewShot_LLM` | Subtle_ReadOnly | 10 | 60.0 | 16.67 | 43.33 | 7.35 |
| `Rule_Only` | Adversarial_Stress | 10 | 0.0 | 0.0 | 0.0 | 2.5 |
| `Rule_Only` | Complex_MultiHop | 10 | 0.0 | 0.0 | 0.0 | 4.5 |
| `Rule_Only` | In_Distribution | 20 | 0.0 | 0.0 | 0.0 | 2.5 |
| `Rule_Only` | Subtle_ReadOnly | 10 | 0.0 | 0.0 | 0.0 | 2.0 |
| `Guarded_Hybrid_Parser` | Adversarial_Stress | 10 | 0.0 | 0.0 | 0.0 | 0.83 |
| `Guarded_Hybrid_Parser` | Complex_MultiHop | 10 | 0.0 | 0.0 | 0.0 | 5.4 |
| `Guarded_Hybrid_Parser` | In_Distribution | 20 | 0.0 | 0.0 | 0.0 | 1.89 |
| `Guarded_Hybrid_Parser` | Subtle_ReadOnly | 10 | 0.0 | 0.0 | 0.0 | 2.0 |

---

## 📄 4. BẢNG MÃ NGUỒN LATEX CHO BÀI BÁO
Đã được xuất tự động tại: [`docs/table_rq3_parser_ablation.tex`](../docs/table_rq3_parser_ablation.tex).

## 📁 5. DỮ LIỆU THÔ
* Toàn bộ 600 bản ghi (50 prompts × 4 cấu hình × 3 lần lặp): `data/processed/scm_results/parser_ablation_benchmark.csv`
* Breakdown theo category: `data/processed/scm_results/parser_ablation_by_category.csv`
