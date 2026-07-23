import os
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

# Import thư viện Gemini của Google
from langchain_google_genai import ChatGoogleGenerativeAI

from architecture_agent import ArchitectureAgent
from performance_agent import PerformanceAgent

# ==========================================
# 1. CẤU HÌNH API KEY VÀ KHỞI TẠO LLM
# ==========================================
# Điền API Key Gemini của bạn vào đây
os.environ["GOOGLE_API_KEY"] = "AIzaSyA5YbGtWaK5vYkFzfPE4z_BjpCycthQf04"

# Khởi tạo mô hình (Nên dùng gemini-1.5-pro cho các bài toán suy luận logic phức tạp)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2) 

# ==========================================
# 2. KHỞI TẠO CÁC AGENT CƠ SỞ
# ==========================================
arch_agent = ArchitectureAgent('C:\\NGUYEN KHANH KY\\NCKH\\mas_architecture_project\\src\\graph\\sockshop_agent_graph.json')
perf_agent = PerformanceAgent('masre-502308') # ID Project BigQuery của bạn

# ==========================================
# 2. ĐỊNH NGHĨA STATE (TRẠNG THÁI TỔNG HỢP)
# ==========================================
class RequirementState(TypedDict):
    input_requirement: str          # Yêu cầu từ người dùng
    core_services: list[str]        # Dịch vụ bị sửa code trực tiếp
    impact_graph: dict              # Sơ đồ tác động API
    performance_metrics: dict       # Dữ liệu tài nguyên hệ thống (CPU, Memory...)
    feasibility_report: str         # Báo cáo đánh giá khả thi cuối cùng

# ==========================================
# 3. ĐỊNH NGHĨA CÁC NODES 
# ==========================================
def extract_core_services_node(state: RequirementState) -> RequirementState:
    """Tác tử 1: Đọc hiểu yêu cầu và tìm dịch vụ cốt lõi (Bản nâng cấp)"""
    print("🔍 [Phase 1] Nhận diện dịch vụ cốt lõi cần sửa đổi...")
    
    # BƯỚC CẢI TIẾN: Trích xuất cả Tên và Mô tả chức năng từ Đồ thị JSON
    services_context = ""
    for node_id in arch_agent.graph.nodes:
        # Lấy mô tả (nếu có, từ file JSON của bạn)
        desc = arch_agent.graph.nodes[node_id].get('description', 'Không có mô tả')
        services_context += f"- {node_id}: {desc}\n"
    
    # Cung cấp bộ quy tắc suy luận rõ ràng cho AI
    prompt = f"""Bạn là một chuyên gia Phân tích Hệ thống (Systems Analyst). Hãy phân tích yêu cầu tính năng mới và chỉ ra CHÍNH XÁC các vi dịch vụ cần được sửa code.
    
    [DANH SÁCH DỊCH VỤ VÀ CHỨC NĂNG HIỆN TẠI]
    {services_context}
    
    [YÊU CẦU TÍNH NĂNG MỚI]
    "{state['input_requirement']}"
    
    [HƯỚNG DẪN SUY LUẬN]
    - Nếu thay đổi Giao diện người dùng (UI) -> Bắt buộc phải có 'front-end'.
    - Nếu thay đổi logic tính toán tiền/giỏ hàng -> Phải có 'carts'.
    - Nếu thay đổi quy trình đặt hàng/thanh toán -> Phải có 'orders' hoặc 'payment'.
    
    CHỈ TRẢ VỀ DUY NHẤT một mảng JSON chứa tên ID của các dịch vụ cốt lõi (Ví dụ định dạng: ["front-end", "carts"]). Tuyệt đối KHÔNG xuất thêm bất kỳ chữ nào khác.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        # Ép kiểu an toàn, loại bỏ các ký tự Markdown thừa nếu LLM tự sinh ra
        core_services = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
    except Exception as e:
        print(f"Lỗi parse JSON: {e} - Raw text: {response.content}")
        core_services = ["front-end"] # Fallback an toàn hơn
        
    print(f"   -> Dịch vụ cốt lõi được AI nhận diện: {core_services}")
    return {"core_services": core_services}

def map_impact_node(state: RequirementState) -> RequirementState:
    """Tác tử 2: Quét đồ thị tìm dịch vụ liên đới"""
    print("🕸️ [Phase 2] Phân tích tác động dây chuyền trên đồ thị...")
    impact_mapping = {}
    for service in state['core_services']:
        impact_mapping[service] = {
            "api_consumers_to_notify": arch_agent.get_upstream_dependencies(service),
            "downstream_services_to_check": arch_agent.get_downstream_dependencies(service)
        }
    print(f"   -> Sơ đồ tác động: Đã quét xong.")
    return {"impact_graph": impact_mapping}

def fetch_performance_node(state: RequirementState) -> RequirementState:
    """Tác tử 3: Kiểm tra sức khỏe tài nguyên hệ thống từ BigQuery"""
    print("📊 [Phase 3] Thu thập dữ liệu hiệu năng thực tế từ BigQuery...")
    
    # Gom tất cả các dịch vụ liên quan (Cốt lõi + Thượng lưu + Hạ lưu) để tránh query trùng
    services_to_check = set(state['core_services'])
    for srv, deps in state['impact_graph'].items():
        services_to_check.update(deps['api_consumers_to_notify'])
        services_to_check.update(deps['downstream_services_to_check'])
        
    collected_metrics = {}
    for srv in services_to_check:
        # Lấy metrics hiện tại (Bạn có thể đổi 'scenario' thành kịch bản bình thường)
        metrics = perf_agent.get_metrics_for_service(srv, scenario='payment_cpu')
        if metrics:
            collected_metrics[srv] = metrics
            
    print(f"   -> Lấy thành công metrics của {len(collected_metrics)} dịch vụ.")
    return {"performance_metrics": collected_metrics}

def generate_report_node(state: RequirementState) -> RequirementState:
    """Tác tử 4: Kiến trúc sư LLM viết báo cáo khả thi toàn diện"""
    print("🧠 [Phase 4] Đang tổng hợp Báo cáo Khả thi (Feasibility Report)...")
    
    system_prompt = """Bạn là Kỹ sư Trưởng (Principal Engineer). Nhiệm vụ của bạn là đánh giá tính khả thi khi thêm tính năng mới vào kiến trúc vi dịch vụ.
    Bạn phải đánh giá trên 2 phương diện:
    1. Tác động kiến trúc (Cần sửa API nào, ai bị ảnh hưởng).
    2. Khả năng chịu tải (Dựa vào metrics, tài nguyên hiện tại có kham nổi logic mới không, hay cần scale server trước)."""
    
    human_prompt = f"""
    - Yêu cầu mới: "{state['input_requirement']}"
    - Dịch vụ bị sửa code chính: {state['core_services']}
    - Các dịch vụ liên đới (Đồ thị): {state['impact_graph']}
    - Hiện trạng tài nguyên (CPU, Memory... từ BigQuery): {state['performance_metrics']}
    
    Hãy viết báo cáo:
    - Đánh giá tác động kiến trúc (Code & API).
    - Phân tích rủi ro tài nguyên (Dịch vụ nào đang có nguy cơ quá tải nếu gánh thêm logic).
    - Kết luận khả thi (CÓ THỂ triển khai ngay / KHÔNG THỂ, cần cấp thêm tài nguyên).
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    response = llm.invoke(messages)
    return {"feasibility_report": response.content}

# ==========================================
# 4. LẮP RÁP WORKFLOW LANGGRAPH
# ==========================================
workflow = StateGraph(RequirementState)

workflow.add_node("extract_core", extract_core_services_node)
workflow.add_node("map_impact", map_impact_node)
workflow.add_node("fetch_performance", fetch_performance_node)
workflow.add_node("generate_report", generate_report_node)

# Thiết lập luồng 4 bước
workflow.set_entry_point("extract_core")
workflow.add_edge("extract_core", "map_impact")
workflow.add_edge("map_impact", "fetch_performance")
workflow.add_edge("fetch_performance", "generate_report")
workflow.add_edge("generate_report", END)

feasibility_analyzer = workflow.compile()

# ==========================================
# 5. THỰC THI KIỂM THỬ
# ==========================================
if __name__ == "__main__":
    print("🚀 KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ KHẢ THI (KIẾN TRÚC + TÀI NGUYÊN)...\n")
    
    initial_state = {
        "input_requirement": "Khách hàng muốn thêm tính năng áp dụng mã giảm giá (Voucher) ở màn hình thanh toán trước khi xác nhận đơn hàng."
    }
    
    result = feasibility_analyzer.invoke(initial_state)
    
    print("\n" + "="*70)
    print("✅ BÁO CÁO ĐÁNH GIÁ TÁC ĐỘNG & KHẢ THI TÀI NGUYÊN:")
    print("="*70)
    print(result['feasibility_report'])