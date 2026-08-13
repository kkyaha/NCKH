# DỰ ÁN DỰ BÁO TẢI VÀ RỦI RO NĂNG LƯỢNG CHO TÍNH NĂNG MỚI BẰNG MÔ HÌNH NHÂN QUẢ SCM
## Structural Causal Models (SCM) & Do-Calculus for Zero-Shot Capacity Planning

Hệ thống dự báo tài nguyên (CPU, Memory, Socket, Latency) và cảnh báo rủi ro quá tải/crash cho **tính năng phần mềm MỚI** bằng Mô Hình Nhân Quả Cấu Trúc (SCM) và phép toán can thiệp $do(Workload)$.

---

## ⚡ Lệnh Chạy Tối Giản (Quickstart)

```bash
# Chạy toàn bộ suite kiểm thử và xuất báo cáo chứng minh
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; python test_f1_rmse.py
```

---

## 📊 Kết Quả Tối Giản (Thư Mục `data/processed/scm_results/`)

Chỉ giữ lại **3 file kết quả cốt lõi duy nhất** để dễ trình bày:

1. 🟢 **[EXECUTIVE_PROOF_DASHBOARD.csv](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/data/processed/scm_results/EXECUTIVE_PROOF_DASHBOARD.csv)**: File Excel tổng hợp chứng minh đầy đủ (F1=1.000, RMSE, p-value < 0.05, Cảnh báo rủi ro).
2. 📊 **[MASTER_SUMMARY.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/data/processed/scm_results/MASTER_SUMMARY.md)**: Master Dashboard dạng Markdown xem ngay 4 bảng tổng hợp.
3. 📈 **[MODEL_COMPARISON.csv](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/data/processed/scm_results/MODEL_COMPARISON.csv)**: Bảng dữ liệu so sánh chi tiết 4 mô hình.

---

## 📖 Thư Mục Tài Liệu (`docs/`)
- 📘 **[TESTING_GUIDE.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/docs/TESTING_GUIDE.md)**: Hướng dẫn quy trình dữ liệu & test cases.
- 🎤 **[PRESENTATION_SLIDES_AND_SCRIPT.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/docs/PRESENTATION_SLIDES_AND_SCRIPT.md)**: Slide thuyết trình 7 phần & kịch bản lời thoại.

---

## 📂 Cấu Trúc Dự Án Tối Giản

```
mas_architecture_project/
├── test_f1_rmse.py                  # 🚀 Lệnh thực thi kiểm thử chính
├── README.md                        # Portal hướng dẫn tổng quan
├── docs/                            # 📖 Thư mục tài liệu (Slide & Guide)
├── data/processed/scm_results/      # 📊 Thư mục kết quả (DUY NHẤT 3 FILE CỐT LÕI)
│   ├── EXECUTIVE_PROOF_DASHBOARD.csv    - File Excel chứng minh tổng hợp 8 dòng
│   ├── MASTER_SUMMARY.md                - Báo cáo Markdown tổng hợp
│   └── MODEL_COMPARISON.csv             - Data so sánh 4 mô hình
└── src/scm/                         # 🧠 Các module mã nguồn core SCM
```
