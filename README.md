# DỰ ÁN DỰ BÁO TẢI VÀ RỦI RO NĂNG LƯỢNG CHO TÍNH NĂNG MỚI BẰNG MÔ HÌNH NHÂN QUẢ SCM
## Structural Causal Models (SCM) & Do-Calculus for Zero-Shot Capacity Planning

Hệ thống dự báo tài nguyên (CPU, Memory, Socket, Latency) và cảnh báo rủi ro quá tải/crash cho **tính năng phần mềm MỚI** bằng Mô Hình Nhân Quả Cấu Trúc (SCM) và phép toán can thiệp $do(Workload)$.

---

## ⚡ 1 LỆNH THỰC THI TỔNG HỢP DUY NHẤT (MASTER SINGLE-COMMAND RUNNER)

```bash
# Chạy toàn bộ 4 bước kiểm thử thực nghiệm (Accuracy P80, 14-Node Graph, Flash Sale, Wilcoxon Test)
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; python run_all_experiments.py
```

---

## 📊 KẾT QUẢ TỔNG HỢP CỐT LÕI (Thư Mục `data/processed/scm_results/`)

Chỉ giữ lại **3 file kết quả cốt lõi duy nhất** để dễ trình bày:

1. 🟢 **[EXECUTIVE_PROOF_DASHBOARD.csv](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/data/processed/scm_results/EXECUTIVE_PROOF_DASHBOARD.csv)**: File Excel tổng hợp chứng minh 3 Tầng đầy đủ (F1, Precision, Recall, p-value < 0.05, Cảnh báo rủi ro).
2. 📊 **[MASTER_SUMMARY.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/data/processed/scm_results/MASTER_SUMMARY.md)**: Master Dashboard dạng Markdown 3 Tầng xem ngay 4 bảng tổng hợp.
3. 📈 **[MODEL_COMPARISON.csv](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/data/processed/scm_results/MODEL_COMPARISON.csv)**: Bảng dữ liệu so sánh chi tiết 4 mô hình.

---

## 📖 THƯ MỤC TÀI LIỆU HƯỚNG DẪN & BÁO CÁO TOÀN DIỆN (`docs/`)
- 📄 **[FULL_SCM_EXPERIMENT_REPORT.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/docs/FULL_SCM_EXPERIMENT_REPORT.md)**: **Báo cáo toàn diện kết quả thực nghiệm NCKH** (Gồm 9 phần đầy đủ tất cả các bảng dữ liệu, 3-tier, 4 mô hình, Wilcoxon test, 3 protocol ablation study, 14-node graph test, và Flash Sale risk alerts).
- 📘 **[TESTING_GUIDE.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/docs/TESTING_GUIDE.md)**: Hướng dẫn quy trình dữ liệu & test cases.
- 🎤 **[PRESENTATION_SLIDES_AND_SCRIPT.md](file:///c:/NGUYEN%20KHANH%20KY/NCKH/mas_architecture_project/docs/PRESENTATION_SLIDES_AND_SCRIPT.md)**: Slide thuyết trình 7 phần & kịch bản lời thoại.

---

## 📂 CẤU TRÚC MÃ NGUỒN TỐI GIẢN & GỌN GÀNG

```
mas_architecture_project/
├── run_all_experiments.py           # 🚀 FILE THỰC THI THỰC NGHIỆM MASTER TỔNG HỢP DUY NHẤT
├── test_f1_rmse.py                  # Module kiểm thử chuẩn F1/RMSE & Tính năng mới
├── test_multi_node_graph.py         # Module kiểm thử trực tiếp Đồ thị 14 Node hợp nhất
├── run_statistical_tests.py         # Module kiểm định ý nghĩa thống kê Wilcoxon Test
├── README.md                        # Portal hướng dẫn tổng quan tối giản
├── docs/                            # 📖 Thư mục tài liệu (Slide & Testing Guide)
│   ├── TESTING_GUIDE.md
│   └── PRESENTATION_SLIDES_AND_SCRIPT.md
├── data/processed/scm_results/      # 📊 Thư mục kết quả (3 FILE CỐT LÕI)
│   ├── EXECUTIVE_PROOF_DASHBOARD.csv    - File Excel chứng minh 3 Tầng
│   ├── MASTER_SUMMARY.md                - Báo cáo Markdown tổng hợp 3 Tầng
│   └── MODEL_COMPARISON.csv             - Data so sánh 4 mô hình
└── src/scm/                         # 🧠 Các module mã nguồn core SCM
    ├── request_router.py                - Phân loại Tiếng Việt & Blast Radius
    ├── scm_pipeline.py                  - Pipeline SCM 4 bước
    ├── multi_metric_evaluator.py        - Đánh giá đa chỉ số tài nguyên
    ├── model_comparison.py              - Benchmark đối chiếu 4 mô hình
    └── universal_scm_predictor.py       - Động cơ dự báo can thiệp do-calculus SCM
```
