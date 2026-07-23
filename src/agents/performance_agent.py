import os
from google.cloud import bigquery

class PerformanceAgent:
    def __init__(self, project_id):
        # Truyền đường dẫn tuyệt đối trỏ tới file JSON bạn vừa tải về
        # (Lưu ý: giữ nguyên chữ 'r' ở trước chuỗi để Python không bị lỗi ký tự gạch chéo trên Windows)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\NGUYEN KHANH KY\NCKH\mas_architecture_project\credentials\bq-key.json"
        
        # Sau khi có "chứng minh thư", tiến hành kết nối
        self.client = bigquery.Client(project=project_id)
        print("✅ [PerformanceAgent] Đã xác thực thành công với Google Cloud!")

    def get_metrics_for_service(self, service_name, scenario='payment_cpu'):
        # ... (Phần code bên dưới giữ nguyên) ...
        query = f"""
            SELECT metric_name, AVG(metric_value) as avg_value
            FROM `mas_architecture.system_metrics_fact`
            WHERE service_name = '{service_name}' 
              AND scenario = '{scenario}'
            GROUP BY metric_name
        """
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Đóng gói thành dictionary để LLM dễ đọc
            metrics = {row.metric_name: round(row.avg_value, 4) for row in results}
            return metrics
        except Exception as e:
            print(f"❌ Lỗi truy vấn BigQuery: {e}")
            return {}