# HƯỚNG DẪN QUY TRÌNH KIỂM THỬ VÀ XỬ LÝ DỮ LIỆU SCM

Tài liệu này chi tiết hóa toàn bộ quy trình xử lý dữ liệu, thiết kế kịch bản kiểm thử (Test Cases), phương pháp gán giá trị can thiệp $do()$ và cách đối chiếu so sánh kết quả kiểm thử trong hệ thống dự báo năng lượng & rủi ro **SCM (Structural Causal Model)** cho kiến trúc microservices.

---

## 1. Quy Trình Xử Lý Dữ Liệu (Data Processing Pipeline)

### 1.1. Nguồn Dữ Liệu Đầu Vào
Dữ liệu đo đạc thực nghiệm lấy từ bộ dữ liệu tiêu chuẩn **RCAEval (SockShop dataset)** trong thư mục `data/raw/`:
* **30 Kịch bản Lỗi (Scenarios)**: Bao gồm 5 microservices mục tiêu (`carts`, `catalogue`, `orders`, `payment`, `user`) kết hợp với 6 loại sự cố (`cpu`, `delay`, `disk`, `loss`, `mem`, `socket`).
* **3 Lần Chạy Lặp (3 Runs)**: Mỗi kịch bản có 3 lần lặp độc lập $\to$ **Tổng cộng 90 runs**.
* **Tệp Tin Dữ Liệu Mỗi Run**:
  * `simple_metrics.csv`: Chứa chuỗi thời gian đo đạc các chỉ số tài nguyên và lưu lượng Workload.
  * `inject_time.txt`: Chứa mốc thời gian UNIX timestamp tại thời điểm sự cố bắt đầu bị bơm vào hệ thống.
  * `cluster_info.json`: Chứa mẫu nhật ký log templates và sơ đồ phụ thuộc mạng microservices.

### 1.2. Trích Xuất Dữ Liệu Giai Đoạn Bình Thường (Normal Phase Filtering)
Để huấn luyện mô hình SCM học mối quan hệ nguyên nhân - kết quả ở trạng thái vận hành ổn định:
1. Đọc mốc thời gian bơm lỗi $t_{\text{inject}}$ từ file `inject_time.txt`.
2. Lọc các bản ghi thỏa mãn điều kiện $t < t_{\text{inject}}$ (khoảng 12 phút hoạt động bình thường đầu tiên của mỗi run).
3. Loại bỏ các giá trị thiếu (`dropna()`) và trích xuất 2 cột: `[Workload, TargetMetric]`.

### 1.3. Phương Pháp Chia Dữ Liệu "Gold Standard"
Không chia dữ liệu ngẫu nhiên hay chia theo mốc thời gian đơn thuần, dữ liệu được chia theo **mức độ tải (Workload Quantile)** để mô phỏng chính xác bài toán *"Thêm yêu cầu mới sẽ làm biến động hệ thống ra sao?"*:
* **Tập Huấn Luyện (Train Set - Bottom 67%)**: Bao gồm các điểm dữ liệu ở dải workload thấp và trung bình ($WL \le P_{67}$).
* **Tập Kiểm Thử / Ground Truth (Test Set - Top 33%)**: Bao gồm các điểm dữ liệu ở dải workload cao ($WL > P_{67}$). Đây chính là **đáp án thực tế đo đạc được** để so sánh với dự báo của SCM.

```
       [-------------- Tập Train (67%) --------------] [--- Tập Test / Ground Truth (33%) ---]
Workload: Min --------------------------------------> P67 ----------------------------------> Max
               (Huấn luyện mô hình SCM)                    (SCM dự báo do(WL) & so sánh ground truth)
```

---

## 2. Tiêu Chuẩn Đánh Giá Và Đo Lường Sai Số

Độ chính xác của mô hình dự báo được đo lường qua 5 chỉ số toán học:

1. **RMSE (Root Mean Squared Error)**: Đo sai số định lượng tuyệt đối giữa dự báo $(\hat{y})$ và thực tế $(y_{\text{true}})$.
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{\text{true}, i} - \hat{y}_i)^2}$$
2. **F1-Score (Phát hiện Rủi ro Quá tải / Overload Risk F1)**: Chuyển đổi chỉ số dự báo thành tín hiệu cảnh báo nhị phân dựa trên ngưỡng quá tải $\tau = \mu_{\text{train}} + 0.5 \cdot \sigma_{\text{train}}$. Đánh giá khả năng nhận diện microservice bị nghẽn.
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **MAE (Mean Absolute Error)**: Sai số tuyệt đối bình quân $\text{MAE} = \frac{1}{N}\sum |y_{\text{true}, i} - \hat{y}_i|$.
4. **MAPE (%) (Mean Absolute Percentage Error)**: Sai số phần trăm tương đối $\text{MAPE} = \frac{100\%}{N}\sum \left|\frac{y_{\text{true}, i} - \hat{y}_i}{y_{\text{true}, i}}\right|$.
5. **R² (Coefficient of Determination)**: Hệ số xác định mức độ phù hợp của mô hình.

---

## 3. Các Test Cases Cụ Thể Và Quy Trình Định Lượng Can Thiệp $do()$

Kịch bản kiểm thử hỗ trợ câu truy vấn tự nhiên bằng **Tiếng Việt** (bao gồm loại bỏ dấu tiếng Việt Unicode qua `unicodedata`):

### 🧪 Test Case 1: `APPLY_PROMO_CODE` (Áp Mã Giảm Giá / Voucher)
* **Câu Truy Vấn Đầu Vào**: `"Áp mã voucher giảm giá 20% khi thanh toán"`
* **Kết Quả Phân Loại**: `APPLY_PROMO_CODE`
* **Blast Radius (Dịch Vụ Ảnh Hưởng)**: `front-end` $\to$ `carts` $\to$ `orders` $\to$ `payment`
* **Định Lượng Can Thiệp $do()$**:
  * Profile tài nguyên: `cpu-heavy` (Giải mã & kiểm tra điều kiện mã giảm giá).
  * Tải bổ sung định lượng: $do(WL_{\text{new}} = WL_{\text{base}} \times 1.20)$ (**+20% Workload**).

### 🧪 Test Case 2: `RECOMMEND_PRODUCTS` (Gợi Ý Sản Phẩm Thông Minh AI)
* **Câu Truy Vấn Đầu Vào**: `"Gợi ý các sản phẩm tất thông minh cho tôi"`
* **Kết Quả Phân Loại**: `RECOMMEND_PRODUCTS`
* **Blast Radius (Dịch Vụ Ảnh Hưởng)**: `front-end` $\to$ `user` $\to$ `catalogue` $\to$ `orders`
* **Định Lượng Can Thiệp $do()$**:
  * Profile tài nguyên: `cpu-memory` (Tính toán ma trận gợi ý & truy vấn DB danh mục).
  * Tải bổ sung định lượng: $do(WL_{\text{new}} = WL_{\text{base}} \times 1.30)$ (**+30% Workload**).

### 🧪 Test Case 3: `TRACK_PACKAGE` (Theo Dõi Hành Trình Đơn Hàng Real-Time)
* **Câu Truy Vấn Đầu Vào**: `"Xem hành trình giao hàng và vị trí đơn hàng real-time"`
* **Kết Quả Phân Loại**: `TRACK_PACKAGE`
* **Blast Radius (Dịch Vụ Ảnh Hưởng)**: `front-end` $\to$ `orders` $\to$ `shipping`
* **Định Lượng Can Thiệp $do()$**:
  * Profile tài nguyên: `socket-latency` (Kết nối polling/websocket xem vị trí vận chuyển).
  * Tải bổ sung định lượng: $do(WL_{\text{new}} = WL_{\text{base}} \times 1.15)$ (**+15% Workload**).

### 🧪 Test Case 4: `WRITE_PRODUCT_REVIEW` (Đánh Giá & Nhận Xét Sản Phẩm)
* **Câu Truy Vấn Đầu Vào**: `"Viết nhận xét đánh giá 5 sao cho sản phẩm"`
* **Kết Quả Phân Loại**: `WRITE_PRODUCT_REVIEW`
* **Blast Radius (Dịch Vụ Ảnh Hưởng)**: `front-end` $\to$ `user` $\to$ `catalogue`
* **Định Lượng Can Thiệp $do()$**:
  * Profile tài nguyên: `disk-memory` (Ghi nhận xét & cập nhật điểm đánh giá).
  * Tải bổ sung định lượng: $do(WL_{\text{new}} = WL_{\text{base}} \times 1.10)$ (**+10% Workload**).

### 🧪 Test Case 5: `PLACE_ORDER` (Đặt Hàng Mua Sản Phẩm)
* **Câu Truy Vấn Đầu Vào**: `"Đặt hàng mua sản phẩm"`
* **Kết Quả Phân Loại**: `PLACE_ORDER`
* **Blast Radius (Dịch Vụ Ảnh Hưởng)**: Tất cả 7 dịch vụ (`front-end`, `user`, `catalogue`, `carts`, `orders`, `payment`, `shipping`).
* **Định Lượng Can Thiệp $do()$**:
  * Profile tài nguyên: `cpu-heavy` (Toàn bộ chuỗi giao dịch nặng).
  * Tải bổ sung định lượng: $do(WL_{\text{new}} = WL_{\text{base}} \times 1.25)$ (**+25% Workload**).

---

## 4. Cách Lấy Kết Quả Và So Sánh Báo Cáo

### 4.1. Thực Thi Lệnh Kiểm Thử Tự Động
Để chạy toàn bộ suite kiểm thử và tạo báo cáo kết quả:
```bash
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; python test_f1_rmse.py
```

### 4.2. Thực Thi So Sánh 4 Mô Hình Benchmark
Để đối chiếu độ sai số SCM với 3 mô hình baseline (`Linear Regression`, `Gradient Boosting`, `Gaussian Process`):
```bash
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; python src/scm/model_comparison.py
```

### 4.3. Đầu Ra Các File Báo Cáo Đã Tạo
Tất cả kết quả kiểm thử được kết xuất tự động vào thư mục `data/processed/scm_results/`:

| Tên File Kết Quả CSV | Nội Dung Chi Tiết |
| :--- | :--- |
| **`01_data_overview.csv`** | Thống kê số lượng mẫu, dải workload Min/Max/Mean/Std của 90 runs RCAEval. |
| **`02_model_accuracy.csv`** | Độ chính xác dự báo (F1-Score, RMSE, MAE, MAPE %, R²) theo từng dịch vụ và chỉ số tài nguyên. |
| **`03_workload_sensitivity.csv`** | Dự báo biến động chỉ số tài nguyên khi workload tăng các mức +10%, +20%, +30%, +50%. |
| **`04_request_impact.csv`** | Tác động toàn hệ thống phân theo từng loại request type. |
| **`05_model_comparison.csv`** | Bang so sánh đối chiếu chỉ số sai số giữa 4 mô hình (LinearReg, GradBoost, GaussProc, SCM). |
| **`test_f1_rmse_evaluation.csv`** | Bảng kiểm thử F1/RMSE cho 5 chỉ số tài nguyên trên 7 dịch vụ microservices. |
| **`test_new_features_simulation.csv`** | Kết quả mô phỏng can thiệp $do(WL)$ cho các tính năng MỚI nhập từ tiếng Việt. |
