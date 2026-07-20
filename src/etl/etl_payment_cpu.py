import pandas as pd
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import bigquery # Đã thêm dòng này

# Nạp các biến môi trường từ file .env
load_dotenv()

def process_metrics_to_fact_table(file_path):
    # [Giữ nguyên nội dung hàm này của bạn]
    print(f"🔄 BƯỚC 1 & 2: Đọc và biến đổi dữ liệu từ {file_path} ...")
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df_melted = pd.melt(df, id_vars=['time'], var_name='raw_metric', value_name='metric_value')
    df_melted[['service_name', 'metric_name']] = df_melted['raw_metric'].str.rsplit('_', n=1, expand=True)
    df_final = df_melted.drop(columns=['raw_metric'])
    df_final.rename(columns={'time': 'timestamp'}, inplace=True)
    df_final = df_final[['timestamp', 'service_name', 'metric_name', 'metric_value']]
    df_final['scenario'] = 'payment_cpu'
    print(f"✅ Transform hoàn tất! Chuẩn bị nạp {df_final.shape[0]} dòng.")
    return df_final

def load_to_bigquery(df):
    """
    LOAD: Đẩy DataFrame lên BigQuery sử dụng Native Client
    """
    print("\n🚀 BƯỚC 3: Khởi tạo luồng nạp dữ liệu lên BigQuery ...")
    
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_name = os.getenv("GCP_DATASET")
    table_name = os.getenv("GCP_TABLE")
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not key_path or not os.path.exists(key_path):
        print(f"❌ Lỗi: Không tìm thấy file khóa JSON tại đường dẫn: {key_path}")
        return

    # Xác thực danh tính
    credentials = service_account.Credentials.from_service_account_file(key_path)
    destination_table = f"{project_id}.{dataset_name}.{table_name}"
    
    # 1. Khởi tạo BigQuery Client
    client = bigquery.Client(credentials=credentials, project=project_id)
    
    # 2. Cấu hình Job nạp dữ liệu
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND", # Tương đương if_exists='append'
    )
    
    print(f"⏳ Đang nạp {len(df)} dòng vào bảng `{destination_table}`...")
    
    try:
        # 3. Kích hoạt Job và đợi kết quả
        job = client.load_table_from_dataframe(df, destination_table, job_config=job_config)
        job.result() # Dòng này sẽ block script cho đến khi BigQuery xử lý xong
        print("✅ THÀNH CÔNG: Toàn bộ dữ liệu đã hạ cánh an toàn trên kho BigQuery!")
    except Exception as e:
        print(f"❌ Nạp dữ liệu thất bại. Lỗi chi tiết: {e}")

# ==========================================
# THỰC THI SCRIPT
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(BASE_DIR, 'data', 'raw', 'payment_cpu', '1', 'simple_metrics.csv')
    
    if os.path.exists(data_path):
        fact_df = process_metrics_to_fact_table(data_path)
        load_to_bigquery(fact_df)
    else:
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {data_path}. Hãy kiểm tra lại cấu trúc thư mục!")