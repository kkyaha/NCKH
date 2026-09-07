# -*- coding: utf-8 -*-
"""
Unit Tests for CapacityAgent (ReAct Pattern)
============================================
Kiểm thử chu trình ReAct của CapacityAgent:
  - Act: Huấn luyện và gọi SCM Dual-Path
  - Observe: Sàng lọc ngưỡng bão hòa
  - Reason & Critique: Đánh giá chuyên gia và tự phản biện Devil's Advocate
"""

import os
import sys
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_TEST_DIR)
sys.path.insert(0, os.path.join(_PROJ_DIR, 'src'))
sys.path.insert(0, os.path.join(_PROJ_DIR, 'src', 'agents'))
sys.path.insert(0, os.path.join(_PROJ_DIR, 'src', 'scm'))

from capacity_agent import CapacityAgent, CapacityAssessment


class DummyLLM:
    """Mock LLM để kiểm thử mà không tốn API call."""
    def invoke(self, messages):
        class Resp:
            content = """{
              "expert_assessment": "Tải lan truyền từ front-end đến carts và orders, suy giảm tại payment.",
              "risk_critique": "[Devil's Advocate]: Khả năng nghẽn hàng đợi tại payment DB nếu lượng đặt hàng tăng đột biến.",
              "recommendations": ["Scale carts replicas từ 2 lên 3.", "Bật Redis cache cho danh mục sản phẩm."]
            }"""
        return Resp()


def test_capacity_agent_initialization():
    """Kiểm tra khởi tạo và huấn luyện SCM Tools."""
    agent = CapacityAgent(llm=DummyLLM(), auto_train=True)
    assert agent._is_trained is True
    assert len(agent.trained_models) > 0
    assert agent.global_dag_model is not None


def test_capacity_agent_backward_compatibility():
    """Kiểm tra tương thích ngược với API cũ."""
    agent = CapacityAgent(llm=DummyLLM(), auto_train=True)
    
    # Fast path per-service
    metrics = agent.get_metrics_for_service('front-end', workload_delta_pct=20.0)
    assert isinstance(metrics, dict)
    assert any('change_pct' in k for k in metrics.keys())

    # Accurate path global dag
    sim_res = agent.simulate_intervention('front-end', delta_pct=25.0, n_samples=50)
    assert isinstance(sim_res, dict)
    assert 'front-end' in sim_res
    assert 'cpu_change_pct' in sim_res['front-end']


def test_capacity_agent_react_cycle():
    """Kiểm tra chu trình ReAct hoàn chỉnh: Act -> Observe -> Reason & Critique."""
    agent = CapacityAgent(llm=DummyLLM(), auto_train=True)
    
    parsed_req = {
        'request_type': 'APPLY_PROMO_CODE',
        'injection_service': 'front-end',
        'injection_delta_pct': 20.0,
        'core_services': ['front-end', 'carts'],
        'affected_services': ['front-end', 'carts', 'orders', 'payment'],
        'confidence': 'HIGH'
    }
    
    impact_graph = {
        'carts': {
            'api_consumers_to_notify': ['front-end'],
            'downstream_services_to_check': []
        }
    }
    
    assessment = agent.assess_capacity(parsed_req, impact_graph)
    
    assert isinstance(assessment, CapacityAssessment)
    assert assessment.status in ('SAFE', 'WARNING', 'CRITICAL')
    assert isinstance(assessment.fast_metrics, dict)
    assert isinstance(assessment.simulation_result, dict)
    assert isinstance(assessment.expert_assessment, str)
    assert "Devil's Advocate" in assessment.risk_critique
    assert len(assessment.recommendations) >= 1
    assert assessment.confidence == 'HIGH'
