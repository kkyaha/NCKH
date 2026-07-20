import pandas as pd

# Chỉnh sửa lại đường dẫn này cho khớp với thư mục thực tế trên máy bạn
file_path = 'C:\\NGUYEN KHANH KY\\NCKH\\mas_architecture_project\\data\\raw\\payment_cpu\\1\\simple_metrics.csv'

try:
    # Đọc file CSV
    df = pd.read_csv(file_path)
    
    print("--- 1. KÍCH THƯỚC DỮ LIỆU ---")
    print(f"Tổng số dòng: {df.shape[0]}, Tổng số cột: {df.shape[1]}")
    
    print("\n--- 2. DANH SÁCH CÁC CỘT (SCHEMA) ---")
    print(df.columns.tolist())
    
    print("\n--- 3. BẢN XEM TRƯỚC (5 DÒNG ĐẦU) ---")
    # Hiển thị 5 dòng đầu tiên để xem cấu trúc lưu trữ
    print(df.head())
    
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file. Hãy kiểm tra lại đường dẫn từ thư mục bạn đang chạy code nhé!")