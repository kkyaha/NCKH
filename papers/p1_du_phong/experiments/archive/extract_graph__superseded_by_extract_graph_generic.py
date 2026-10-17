import yaml
import json
import re

def extract_and_generate(compose_file, output_json, output_mermaid):
    print(f"🔄 Đang đọc cấu trúc từ {compose_file}...")
    
    with open(compose_file, 'r', encoding='utf-8') as file:
        compose_data = yaml.safe_load(file)

    services = compose_data.get('services', {})
    nodes = []
    edges = []
    
    # 1. Duyệt qua để lấy danh sách Nodes (Các dịch vụ)
    for srv_name, config in services.items():
        # Phân loại tự động dựa trên tên
        node_type = "database" if "-db" in srv_name or srv_name in ["rabbitmq", "session-db"] else "backend"
        if srv_name == "front-end": 
            node_type = "gateway"
            
        nodes.append({"id": srv_name, "type": node_type})
        
        # 2. Xử lý siêu linh hoạt biến môi trường (Bao trọn List và Dictionary)
        envs = config.get('environment', [])
        env_values = []
        
        if isinstance(envs, list):
            # Lọc bỏ các giá trị rỗng và chuyển thành chuỗi
            env_values = [str(e) for e in envs if e is not None]
        elif isinstance(envs, dict):
            # Nếu là dạng Dict, rút trích các giá trị (values)
            env_values = [str(v) for v in envs.values() if v is not None]
            
        for env in env_values:
            # Dùng Regex quét rộng: Tìm bất kỳ chuỗi nào có dạng "://tên-dịch-vụ"
            # Hỗ trợ tóm gọn cả http, tcp, mongodb, mysql, amqp...
            match = re.search(r'://([a-zA-Z0-9_-]+)', env)
            if match:
                target = match.group(1)
                
                # Xác nhận target có thực sự là một dịch vụ trong hệ thống
                if target in services and target != srv_name:
                    # Chống lặp cạnh (Duplicate Edges)
                    if not any(e['source'] == srv_name and e['target'] == target for e in edges):
                        edges.append({
                            "source": srv_name,
                            "target": target,
                            "relation": "NETWORK_CALL"
                        })
                        
        # 3. Phân tích bổ sung qua depends_on
        depends = config.get('depends_on', [])
        # Docker Compose v3 có thể khai báo depends_on dạng dict
        if isinstance(depends, dict):
            depends = list(depends.keys())
            
        if isinstance(depends, list):
            for dep in depends:
                # Nếu chưa có cạnh NETWORK_CALL, bổ sung cạnh DEPENDS_ON
                if not any(e['source'] == srv_name and e['target'] == dep for e in edges):
                    edges.append({
                        "source": srv_name,
                        "target": dep,
                        "relation": "DEPENDS_ON"
                    })

    # ==========================================
    # KẾT XUẤT 1: Tạo file JSON
    # ==========================================
    graph_data = {"nodes": nodes, "edges": edges}
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4, ensure_ascii=False)
    print(f"✅ Đã tạo file AI Agent: {output_json} (Ghi nhận được {len(edges)} cạnh kết nối!)")

    # ==========================================
    # KẾT XUẤT 2: Tạo mã Mermaid
    # ==========================================
    with open(output_mermaid, 'w', encoding='utf-8') as f:
        f.write("graph TD\n") 
        for node in nodes:
            if node['type'] == 'database':
                f.write(f"    {node['id']}[({node['id']})]\n") 
            elif node['type'] == 'gateway':
                f.write(f"    {node['id']}{{ {node['id']} }}\n") 
            else:
                f.write(f"    {node['id']}\n")
                
        f.write("\n")
        for edge in edges:
            f.write(f"    {edge['source']} -->|{edge['relation']}| {edge['target']}\n")
            
    print(f"✅ Đã tạo mã Sơ đồ trực quan: {output_mermaid}")

if __name__ == "__main__":
    # Đảm bảo bạn đang trỏ đúng đường dẫn tuyệt đối hoặc tương đối tới file YAML
    extract_and_generate(
        compose_file=r'mas_architecture_project/src/etl/docker-compose.yml',
        output_json='sockshop_agent_graph.json',
        output_mermaid='sockshop_diagram.mmd'
    )