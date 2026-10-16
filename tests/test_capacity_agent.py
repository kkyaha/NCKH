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
import numpy as np
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


# ----------------------------------------------------------------------------
# He A song chuyen sang SS-TRAIN (tai THAT, khong phai RCAEval) -- xem
# papers/p1_du_phong/experiments/model_eval/evaluate_sstrain.py ve ly do doi.
# ----------------------------------------------------------------------------
_SSTRAIN_DIR = os.path.join(_PROJ_DIR, 'data', 'raw', 'SS-TRAIN')
_SSTRAIN_OK = os.path.isdir(_SSTRAIN_DIR)
pytestmark_sstrain = pytest.mark.skipif(not _SSTRAIN_OK, reason='can data/raw/SS-TRAIN (khong theo git)')


@pytestmark_sstrain
def test_capacity_agent_trains_on_sstrain():
    """Train tren SS-TRAIN phai thanh cong (khong am tham roi vao nhanh
    'khong load duoc du lieu da dich vu' cua train_accurate_path())."""
    agent = CapacityAgent(llm=DummyLLM(), data_dir=_SSTRAIN_DIR, auto_train=True)
    assert agent._is_trained is True
    assert len(agent.trained_models) > 0
    assert agent.global_dag_model is not None
    assert agent.df_baseline is not None and len(agent.df_baseline) > 0


@pytestmark_sstrain
def test_orchestrator_capacity_agent_uses_sstrain():
    """He A song trong orchestrator.py phai tro dung SS-TRAIN, khong con
    roi ve RE2-SS (RCAEval)."""
    orch = pytest.importorskip('orchestrator', reason='orchestrator can langgraph/langchain')
    assert 'SS-TRAIN' in orch.capacity_agent.data_dir
    assert orch.capacity_agent._is_trained is True


# ----------------------------------------------------------------------------
# Correction chain-awareness trong assess_capacity() -- xem
# papers/p1_du_phong/experiments/model_eval/evaluate_chain_awareness.py
# (994 diem tu SS-LIMITS/SS-LIMITS-CLEAN, Wilcoxon p=0.0234, r=0.889).
# ----------------------------------------------------------------------------
_CHAIN_CORRECTION_PATH = os.path.join(_PROJ_DIR, 'data', 'processed', 'frozen',
                                       'chain_correction_params.json')
pytestmark_chaincorr = pytest.mark.skipif(
    not (_SSTRAIN_OK and os.path.isfile(_CHAIN_CORRECTION_PATH)),
    reason='can data/raw/SS-TRAIN + data/processed/frozen/chain_correction_params.json')


@pytestmark_sstrain
def test_capacity_agent_computes_rho_on_sstrain():
    """rho_s (ti le workload/gateway) phai tinh duoc cho moi service sau khi
    train tren SS-TRAIN, va rho cua gateway voi chinh no phai = 1.0."""
    agent = CapacityAgent(llm=DummyLLM(), data_dir=_SSTRAIN_DIR, auto_train=True)
    assert agent.rho
    assert agent.rho[agent.default_injection] == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= v for v in agent.rho.values())


@pytestmark_chaincorr
def test_apply_chain_correction_masks_out_of_chain_and_scales_in_chain():
    """Service NGOAI chain phai bi ep ve cpu_predicted=cpu_baseline (delta=0);
    service TRONG chain phai duoc nhan dung he so suy tu log_a/b da nap."""
    agent = CapacityAgent(llm=DummyLLM(), data_dir=_SSTRAIN_DIR, auto_train=True)
    assert agent._chain_correction is not None

    sim_result = {
        'carts':   {'cpu_baseline': 10.0, 'cpu_predicted': 12.0, 'cpu_change_pct': 20.0},
        'payment': {'cpu_baseline': 1.0,  'cpu_predicted': 1.3,  'cpu_change_pct': 30.0},
    }
    out = agent._apply_chain_correction(sim_result, chain={'carts'})

    # Ngoai chain: ep ve baseline, danh dau da sua.
    assert out['payment']['cpu_predicted'] == pytest.approx(1.0)
    assert out['payment']['cpu_change_pct'] == 0.0
    assert out['payment']['chain_corrected'] is True

    # Trong chain: delta goc (12.0-10.0=2.0) nhan voi he so suy tu rho['carts'].
    log_a, b = agent._chain_correction['log_a'], agent._chain_correction['b']
    rho_carts = agent.rho['carts']
    expected_ratio = np.exp(log_a + b * np.log(1.0 / rho_carts))
    expected_pred = 10.0 + 2.0 * expected_ratio
    assert out['carts']['cpu_predicted'] == pytest.approx(expected_pred)
    assert out['carts']['chain_corrected'] is True


@pytestmark_chaincorr
def test_apply_chain_correction_noop_when_chain_empty_or_no_calibration():
    """Chain rong -> tra nguyen sim_result. Thieu calibration (_chain_correction=None)
    -> cung tra nguyen, khong raise -- dam bao fallback an toan khi thieu file."""
    agent = CapacityAgent(llm=DummyLLM(), data_dir=_SSTRAIN_DIR, auto_train=True)
    sim_result = {'carts': {'cpu_baseline': 10.0, 'cpu_predicted': 12.0, 'cpu_change_pct': 20.0}}

    assert agent._apply_chain_correction(sim_result, chain=set()) == sim_result

    agent._chain_correction = None
    assert agent._apply_chain_correction(sim_result, chain={'carts'}) == sim_result


@pytestmark_chaincorr
def test_assess_capacity_applies_chain_correction_end_to_end():
    """assess_capacity() voi affected_services hep phai cho ra simulation_result
    co service ngoai chain bi mask (cpu_change_pct=0, chain_corrected=True)."""
    agent = CapacityAgent(llm=DummyLLM(), data_dir=_SSTRAIN_DIR, auto_train=True)
    parsed_req = {
        'request_type': 'GET_CATALOGUE', 'injection_service': 'front-end',
        'injection_delta_pct': 20.0, 'core_services': ['front-end', 'catalogue'],
        'affected_services': ['front-end', 'catalogue'], 'confidence': 'HIGH',
    }
    assessment = agent.assess_capacity(parsed_req, impact_graph=None)
    sim = assessment.simulation_result
    for svc in ('orders', 'payment', 'shipping'):
        if svc in sim and sim[svc].get('cpu_predicted') not in (None, float('inf')):
            assert sim[svc].get('chain_corrected') is True
            assert sim[svc]['cpu_change_pct'] == 0.0
