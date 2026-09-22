# -*- coding: utf-8 -*-
"""Kiem thu NewFeatureFeasibilityAgent (src/agents/feasibility_agent.py): C_s song, bootstrap, assess()."""

import json
import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'agents'))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'scm'))

import feasibility_agent as FA  # noqa: E402
import feasibility_predictor as FP  # noqa: E402

FROZEN_OK = os.path.exists(FA.DEFAULT_FROZEN) and os.path.exists(FA.DEFAULT_P2)
TRAIN_OK = os.path.isdir(FA.DEFAULT_TRAIN_DIR)
pytestmark = pytest.mark.skipif(not FROZEN_OK, reason='can data/processed/frozen/predictions_frozen_RE2.json + p2_params_dev.json')


# ---------------- read_live_cores: khong can Docker that (mock subprocess) ----------------
class _FakeProc:
    def __init__(self, out='', code=0):
        self.stdout, self.returncode, self.stderr = out, code, ''


def test_read_live_cores_success(monkeypatch):
    def fake_run(cmd, **kw):
        return _FakeProc(str(50_000_000 if 'user' in cmd[2] else 500_000_000))
    monkeypatch.setattr(FA.subprocess, 'run', fake_run)
    cores, src, warn = FA.read_live_cores(['front-end', 'user'])
    assert cores == {'front-end': 0.5, 'user': 0.05} and src == 'docker (song)' and warn == []


def test_read_live_cores_falls_back_when_docker_unavailable(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        raise FileNotFoundError('docker khong ton tai')
    monkeypatch.setattr(FA.subprocess, 'run', fake_run)
    lim = tmp_path / 'limits.json'
    lim.write_text(json.dumps({'configs': {'RE2': {'front-end': 0.5}}}), encoding='utf-8')
    cores, src, warn = FA.read_live_cores(['front-end'], fallback_limits=str(lim))
    assert cores == {'front-end': 0.5} and 'du phong' in src and len(warn) == 1


def test_read_live_cores_unlimited_nanocpus_zero(monkeypatch):
    """NanoCpus=0 (chua bao gio bi dat tran) -> quy ve NCPU cua may, khong phai 0 core."""
    def fake_run(cmd, **kw):
        if 'NCPU' in cmd[-1]:
            return _FakeProc('8')
        return _FakeProc('0')
    monkeypatch.setattr(FA.subprocess, 'run', fake_run)
    cores, src, warn = FA.read_live_cores(['front-end'])
    assert cores['front-end'] == 8.0 and warn == []


# ---------------- assess(): dung du lieu dong bang that ----------------
@pytest.fixture
def agent(monkeypatch):
    def fake_cores(services, *a, **kw):
        return {s: 0.5 if s not in ('user', 'payment') else (0.2 if s == 'user' else 0.1) for s in services}, 'gia lap', []
    monkeypatch.setattr(FA, 'read_live_cores', fake_cores)
    return FA.NewFeatureFeasibilityAgent()


def test_assess_point_estimate_matches_frozen_prediction(agent):
    v = agent.assess('promo', scale=1.0, L_peak=100, bootstrap=False)
    frozen_p1 = next(p for p in agent.frozen['predictions'] if p['predictor'] == 'P1' and p['feature'] == 'promo' and p['scale'] == 1.0)
    # cores gia lap khac RE2 that nen khong doi khop tuyet doi; kiem tinh hop ly: P2 (co chi phi dieu
    # phoi gateway) PHAI gay som hon hoac bang P1 (dung ket luan da ghi trong docs muc 5c/5h), va cung bac do lon.
    assert 50 < v.breakpoint_rps <= frozen_p1['breakpoint_rps']
    assert v.model == 'P2' and v.k_source == 'gia dinh (k=1)'


def test_assess_verdict_bands(agent):
    below = agent.assess('promo', scale=1.0, L_peak=10, bootstrap=False)
    above = agent.assess('promo', scale=1.0, L_peak=400, bootstrap=False)
    assert below.verdict == 'FEASIBLE' and above.verdict == 'INFEASIBLE'
    assert below.max_utilization < above.max_utilization


def test_assess_latency_risk_flag_high_fanout(agent):
    k_express = {'catalogue': 1.0, 'user': 6.0, 'carts': 3.0, 'orders': 1.0, 'payment': 1.0, 'shipping': 1.0}
    v = agent.assess('express', scale=1.0, L_peak=50, k=k_express, bootstrap=False)
    assert v.latency_risk is True and v.model == 'P3' and v.k_source == 'do bang probe'
    v_light = agent.assess('promo', scale=1.0, L_peak=50, bootstrap=False)
    assert v_light.latency_risk is False


def test_assess_ci_width_zero_without_bootstrap(agent):
    v = agent.assess('recs', scale=1.0, L_peak=100, bootstrap=False)
    assert v.breakpoint_ci == (v.breakpoint_rps, v.breakpoint_rps) and v.n_bootstrap == 0


@pytest.mark.skipif(not TRAIN_OK, reason='can data/raw/SS-TRAIN de bootstrap')
def test_bootstrap_ci_contains_point_estimate_and_shrinks_with_more_data(agent):
    v = agent.assess('recs', scale=1.0, L_peak=100, bootstrap=True, n_bootstrap=60, seed=1)
    lo, hi = v.breakpoint_ci
    assert lo <= v.breakpoint_rps <= hi and hi > lo   # khoang thuc su co do rong (khong suy bien)
    assert v.n_bootstrap == 60


@pytest.mark.skipif(not TRAIN_OK, reason='can data/raw/SS-TRAIN de bootstrap')
def test_bootstrap_is_deterministic_given_seed(agent):
    v1 = agent.assess('recs', scale=1.0, L_peak=100, bootstrap=True, n_bootstrap=40, seed=7)
    v2 = agent.assess('recs', scale=1.0, L_peak=100, bootstrap=True, n_bootstrap=40, seed=7)
    assert v1.breakpoint_ci == v2.breakpoint_ci


def test_assess_without_train_dir_warns_instead_of_crashing(monkeypatch, agent):
    agent.train_dir = '/khong/ton/tai'
    v = agent.assess('recs', scale=1.0, L_peak=100, bootstrap=True, n_bootstrap=10)
    assert v.n_bootstrap == 0 and any('khong ton tai' in w for w in v.warnings)
