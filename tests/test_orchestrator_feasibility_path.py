# -*- coding: utf-8 -*-
"""Chot DUONG DI cua cau hoi "tinh nang moi co kha thi khong?" trong orchestrator.

Vi sao can test nay: `docs/DATA_FRAMEWORK.md` muc 2 ghi 4 diem ma LECH KHUNG, tat ca nam trong
duong CU (`CapacityAgent` + StateGraph, dung cho RQ1-RQ4):
  * capacity_agent.py:724   baseline = trung binh GOP (khung: baseline tai L_peak)
  * capacity_agent.py:1351  tran = P99 cua train (khung: C_s that tu telemetry)
  * parser_agent.py:451     Delta = prior cua archetype (khung: them mot LOP request lambda*)
  * assess_capacity:1491    chain chi la mat na chon pham vi (khung: W_s' = W_s + lambda*·k_s)
Cach giai quyet KHONG phai va lai `CapacityAgent` (se pha cac so RQ1-RQ4 da bao cao) ma la: co che
dung khung duoc cai thanh mot thanh phan RIENG (`feasibility_predictor` + `NewFeatureFeasibilityAgent`)
va duong vao cau hoi kha thi tro toi do. Test nay chot dieu do de no khong am tham quay lai duong cu.
"""

import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'agents'))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'scm'))

import feasibility_agent as FA  # noqa: E402

FROZEN_OK = os.path.exists(FA.DEFAULT_FROZEN) and os.path.exists(FA.DEFAULT_P2)
pytestmark = pytest.mark.skipif(
    not FROZEN_OK, reason='can data/processed/frozen/predictions_frozen_RE2.json + p2_params_dev.json')


@pytest.fixture(scope='module')
def orch():
    return pytest.importorskip('orchestrator', reason='orchestrator can langgraph/langchain')


@pytest.fixture(scope='module')
def FA_orch(orch):
    """Module feasibility_agent DUNG OBJECT ma orchestrator nap.

    Orchestrator nap `agents.feasibility_agent` (duong package) con test nay nap
    `feasibility_agent` (duong module) -> Python coi la HAI module khac nhau, nen
    `isinstance` va monkeypatch tren ban nay KHONG tac dung len ban kia. Lay dung
    ban cua orchestrator de test chot hanh vi THAT."""
    return sys.modules[orch._get_feasibility_agent().__class__.__module__]


def _fake_parsed(request_type, delta_pct):
    """ParsedRequirement toi thieu: khong goi LLM, khong can mang."""
    import parser_agent as PA
    kw = {f.name: (0.0 if f.type is float else ('' if f.type is str else []))
          for f in PA.ParsedRequirement.__dataclass_fields__.values()}
    kw.update(request_type=request_type, injection_service='front-end',
              injection_delta_pct=delta_pct, core_services=['front-end'],
              affected_services=['front-end'], reasoning='test', confidence='HIGH',
              matched_template=request_type, template_delta=delta_pct, adjustment=0.0,
              similarity_score=1.0, llm_was_called=False)
    return PA.ParsedRequirement(**{k: v for k, v in kw.items()
                                   if k in PA.ParsedRequirement.__dataclass_fields__})


def test_new_feature_path_uses_framework_predictor_not_capacity_agent(orch, FA_orch, monkeypatch):
    """Duong tinh nang moi phai tra FeasibilityVerdict cua P2/P3, KHONG goi CapacityAgent."""
    monkeypatch.setattr(orch.parser_agent, 'parse', lambda text: _fake_parsed('LOGIN', 10.0))

    called = []
    if hasattr(orch, 'capacity_agent') and hasattr(orch.capacity_agent, 'CapacityAgent'):
        monkeypatch.setattr(orch.capacity_agent.CapacityAgent, 'assess_capacity',
                            lambda *a, **kw: called.append('capacity') or {})

    out = orch.assess_new_feature_requirement('cho phep khach dang nhap', L_peak=150.0, bootstrap=False)
    v = out['feasibility']
    assert v is not None and out.get('error') is None
    assert isinstance(v, FA_orch.FeasibilityVerdict)
    assert v.model in ('P2', 'P3')
    assert v.verdict in ('FEASIBLE', 'MARGINAL', 'INFEASIBLE')
    assert called == [], 'duong tinh nang moi KHONG duoc goi CapacityAgent.assess_capacity'


def test_new_feature_path_matches_agent_called_directly(orch, FA_orch, monkeypatch):
    """Khong co lop tinh toan nao khac chen vao: ket qua trung khop voi goi agent truc tiep."""
    monkeypatch.setattr(orch.parser_agent, 'parse', lambda text: _fake_parsed('LOGIN', 10.0))
    out = orch.assess_new_feature_requirement('dang nhap', L_peak=150.0, bootstrap=False)
    direct = FA_orch.NewFeatureFeasibilityAgent().assess('LOGIN', scale=1.0, L_peak=150.0, bootstrap=False)
    assert out['feasibility'].breakpoint_rps == direct.breakpoint_rps
    assert out['feasibility'].bottleneck == direct.bottleneck


def test_scale_follows_parsed_delta_not_archetype_anchor(orch, FA_orch, monkeypatch):
    """Delta rieng cua yeu cau (20%) phai duoc ton trong, khong bi thay bang anchor cua archetype (10%)."""
    from feasibility_predictor import SOCKSHOP_CALL_CHAINS
    anchor = SOCKSHOP_CALL_CHAINS['LOGIN']['expected_delta_pct']
    assert anchor == 10, 'test nay gia dinh anchor cua LOGIN = 10%'

    seen = {}
    real = FA_orch.NewFeatureFeasibilityAgent.assess

    def spy(self, feature, scale=1.0, **kw):
        seen['scale'] = scale
        return real(self, feature, scale=scale, **kw)
    monkeypatch.setattr(FA_orch.NewFeatureFeasibilityAgent, 'assess', spy)
    orch._feasibility_agent = None      # buoc tao lai bang class da bi patch
    monkeypatch.setattr(orch.parser_agent, 'parse', lambda text: _fake_parsed('LOGIN', 2 * anchor))
    orch.assess_new_feature_requirement('dang nhap, tai gap doi', L_peak=150.0, bootstrap=False)
    assert seen['scale'] == pytest.approx(2.0)


def test_out_of_taxonomy_archetype_is_refused_not_guessed(orch, monkeypatch):
    """Ngoai taxonomy -> tra loi RO rang la ngoai pham vi, khong doan mot con so."""
    monkeypatch.setattr(orch.parser_agent, 'parse', lambda text: _fake_parsed('KHONG_CO_TRONG_TAXONOMY', 10.0))
    out = orch.assess_new_feature_requirement('yeu cau la', L_peak=150.0, bootstrap=False)
    assert out['feasibility'] is None and 'ngoai pham vi' in out['error']
