import json
import networkx as nx

class ArchitectureAgent:
    def __init__(self, graph_json_path):
        self.graph = nx.DiGraph()
        self._load_graph(graph_json_path)

    def _load_graph(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for node in data['nodes']:
            self.graph.add_node(node['id'], type=node['type'])
            
        for edge in data['edges']:
            self.graph.add_edge(edge['source'], edge['target'], relation=edge['relation'])
            
    def get_downstream_dependencies(self, service_name):
        """Trả về danh sách các dịch vụ bị gọi (hạ lưu)"""
        if service_name not in self.graph:
            return []
        return list(self.graph.successors(service_name))
        
    def get_upstream_dependencies(self, service_name):
        """Trả về danh sách các dịch vụ gọi đến (thượng lưu) - Hữu ích để tìm ai làm sập mình"""
        if service_name not in self.graph:
            return []
        return list(self.graph.predecessors(service_name))