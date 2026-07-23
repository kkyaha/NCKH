import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from rca_agents import ArchitectureAgent, DataAgent # Import 2 class đã viết ở trên

# ==========================================
# 1. ĐỊNH NGHĨA STATE (TRẠNG THÁI DÙNG CHUNG)
# ==========================================
class RCAState(TypedDict):
    alert_service: str                  # Đầu vào: Dịch vụ đang báo lỗi
    suspects: list[str]                 # Danh sách các dịch vụ hạ lưu bị tình nghi
    metrics_data: dict                  # Dữ liệu đo lường lấy từ BigQuery
    final_root_cause: str               # Đầu ra: Kết luận nguyên nhân gốc rễ

# ==========================================
# 2. KHỞI TẠO CÁC TÁC TỬ CƠ SỞ
# ==========================================
arch_agent = ArchitectureAgent('sockshop_architecture_graph.json')
data_agent = DataAgent('your-gcp-project-id') # Nhớ thay ID dự án BigQuery

# ==========================================
# 3. ĐỊNH NGHĨA CÁC NODES (TÁC TỬ TRONG LANGGRAPH)
# ==========================================
def architecture_node(state: RCAState) -> RCAState:
    """Node 1: Phân tích kiến trúc để tìm nghi phạm"""
    print(f"🕵️ [Arch Node] Phân tích sự phụ thuộc của: {state['alert_service']}")
    suspects = arch_agent.get_downstream_dependencies(state['alert_service'])
    print(f"   -> Đã khoanh vùng nghi phạm: {suspects}")
    return {"suspects": suspects} # Cập nhật State

def data_node(state: RCAState) -> RCAState:
    """Node 2: Lấy dữ liệu BigQuery cho các nghi phạm"""
    print(f"📊 [Data Node] Đang truy vấn BigQuery cho các nghi phạm...")
    collected_metrics = {}
    for suspect in state['suspects']:
        # Fetch metrics từ BigQuery
        metrics = data_agent.get_metrics_for_service(suspect)
        collected_metrics[suspect] = metrics
        print(f"   -> Lấy thành công dữ liệu của {suspect}")
        
    return {"metrics_data": collected_metrics} # Cập nhật State

def llm_node(state: RCAState) -> RCAState:
    """Node 3: Khối óc LLM đánh giá bằng chứng và kết luận"""
    print(f"🧠 [LLM Node] Đang phân tích dữ liệu và suy luận...")
    
    # Ở đây chúng ta tạm mô phỏng tư duy của LLM
    # Trong thực tế, bạn sẽ đưa state['metrics_data'] vào prompt của OpenAI/Gemini
    metrics = state['metrics_data']
    
    # Giả lập logic: LLM phát hiện payment có CPU bất thường
    root_cause = f"Dựa trên dữ liệu, dịch vụ gây lỗi gốc là 'payment' do quá tải tài nguyên. Các dịch vụ {state['suspects']} chỉ là nạn nhân dây chuyền."
    
    print(f"   -> Kết luận: {root_cause}")
    return {"final_root_cause": root_cause}

# ==========================================
# 4. XÂY DỰNG ĐỒ THỊ ĐIỀU PHỐI (WORKFLOW GRAPH)
# ==========================================
workflow = StateGraph(RCAState)

# Thêm các Node vào đồ thị
workflow.add_node("architecture_analyzer", architecture_node)
workflow.add_node("data_fetcher", data_node)
workflow.add_node("llm_reasoner", llm_node)

# Thiết lập điểm bắt đầu
workflow.set_entry_point("architecture_analyzer")

# Nối các Cạnh (Edges) để tạo luồng chảy
workflow.add_edge("architecture_analyzer", "data_fetcher")
workflow.add_edge("data_fetcher", "llm_reasoner")
workflow.add_edge("llm_reasoner", END)

# Biên dịch đồ thị thành ứng dụng chạy được
rca_app = workflow.compile()

# ==========================================
# THỰC THI KIỂM THỬ
# ==========================================
if __name__ == "__main__":
    print("\n🚀 KHỞI ĐỘNG HỆ THỐNG LANGGRAPH RCA...\n")
    
    # Kích hoạt hệ thống với input ban đầu
    initial_state = {"alert_service": "front-end"}
    
    # Chạy đồ thị
    final_state = rca_app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("✅ BÁO CÁO RCA HOÀN CHỈNH")
    print("="*50)
    print(final_state['final_root_cause'])