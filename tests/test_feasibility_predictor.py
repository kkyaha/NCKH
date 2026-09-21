# -*- coding: utf-8 -*-
"""Kiem thu bo du doan kha thi P0/P1 (src/scm/feasibility_predictor.py) tren du lieu tong hop."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'scm'))
sys.path.insert(0, os.path.join(_PROJ, 'experiments'))

import feasibility_predictor as FP  # noqa: E402

RHO = {'front-end': 1.0, 'catalogue': 0.57, 'user': 0.33, 'carts': 0.41, 'orders': 0.035, 'payment': 0.035, 'shipping': 0.035}
A = {'front-end': 0.3, 'catalogue': 0.2, 'user': 0.2, 'carts': 0.5, 'orders': 0.2, 'payment': 0.05, 'shipping': 0.1}
B = {'front-end': 0.2, 'catalogue': 0.055, 'user': 0.07, 'carts': 0.17, 'orders': 0.8, 'payment': 0.04, 'shipping': 0.3}
CORES = {s: 0.5 for s in FP.SERVICES}
CORES.update(user=0.2, payment=0.1)


def synth(n=600, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    wg = rng.uniform(20, 150, n)
    d = {'front-end_workload': wg, 'load_level': (wg // 25) * 25}
    for s in FP.SERVICES:
        w = RHO[s] * wg * (1 + rng.normal(0, noise, n)) if s != 'front-end' else wg
        d[f'{s}_workload'] = w
        d[f'{s}_cpu'] = A[s] + B[s] * w + rng.normal(0, 0.05, n)
    return pd.DataFrame(d)


@pytest.fixture(scope='module')
def pred():
    return FP.FeasibilityPredictor(FP.fit_mechanism(synth()), CORES, u_star=0.87)


def test_mechanism_recovers_coefficients():
    m = FP.fit_mechanism(synth())
    for s in FP.SERVICES:
        assert m[s]['rho'] == pytest.approx(RHO[s], rel=0.03), s
        assert m[s]['beta'] == pytest.approx(B[s], rel=0.05), s
        if s in FP.SCORED:      # payment/shipping: CPU qua nho so voi nhieu -> R2 thap la dung, va khong duoc cham diem
            assert m[s]['r2'] > 0.9, s


@pytest.mark.parametrize('mode,feature,scale', [('P1', None, 1), ('P0', 'promo', 1), ('P1', 'promo', 2),
                                                 ('P1', 'recs', 1), ('P0', 'recs', 2), ('P1', 'track', 1)])
def test_closed_form_breakpoint_matches_scan(pred, mode, feature, scale):
    rstar, node = pred.breakpoint(mode=mode, feature=feature, scale=scale)
    just_below = pred.verdict(rstar * 0.999, mode=mode, feature=feature, scale=scale)
    just_above = pred.verdict(rstar * 1.001, mode=mode, feature=feature, scale=scale)
    assert just_below['verdict'] != 'INFEASIBLE' and just_above['verdict'] == 'INFEASIBLE'
    assert just_above['bottleneck'] == node


def test_p0_spreads_load_to_all_services_p1_only_to_chain(pred):
    base = pred.workloads(100, mode='P1')
    p0 = pred.workloads(100, mode='P0', feature='promo')
    p1 = pred.workloads(100, mode='P1', feature='promo')            # chain promo: front-end,carts,orders,payment
    assert p0['catalogue'] == pytest.approx(base['catalogue'] * 1.2)   # P0: moi service +20%
    assert p1['catalogue'] == pytest.approx(base['catalogue'])         # P1: ngoai chain khong doi
    assert p1['orders'] == pytest.approx(base['orders'] + 20.0)        # P1: +Delta*L*k = +20 req/s
    assert p1['front-end'] == pytest.approx(120.0)                     # gateway thay TONG request


def test_p1_predicts_earlier_breakpoint_for_heavy_orders_feature(pred):
    r0, _ = pred.breakpoint(mode='P0', feature='recs', scale=2)
    r1, node = pred.breakpoint(mode='P1', feature='recs', scale=2)
    assert r1 < r0 and node in ('orders', 'catalogue', 'front-end')


def test_scale_monotone_and_base_is_latest(pred):
    base = pred.breakpoint(mode='P1')[0]
    r1 = pred.breakpoint(mode='P1', feature='promo', scale=1)[0]
    r2 = pred.breakpoint(mode='P1', feature='promo', scale=2)[0]
    assert r2 < r1 < base


def test_verdict_bands(pred):
    r, _ = pred.breakpoint(mode='P1')
    assert pred.verdict(r * 0.5)['verdict'] == 'FEASIBLE'
    assert pred.verdict(r * 0.95)['verdict'] == 'MARGINAL'
    assert pred.verdict(r * 1.05)['verdict'] == 'INFEASIBLE'


def test_wrong_chain_is_same_size_has_gateway_deterministic():
    for f in FP.FEATURE_ARCHETYPE:
        true = FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE[f]]['services']
        w = FP.wrong_chain(f, 0)
        assert len(w) == len(true) and w[0] == FP.GATEWAY and len(set(w)) == len(w)
        assert w == FP.wrong_chain(f, 0)


def test_select_train_rejects_feature_rows_and_limits():
    d = synth()
    d['feature'] = 'base'
    assert len(FP.select_train(d, 150)) == len(d)
    d.loc[0, 'feature'] = 'promo'
    with pytest.raises(ValueError, match='ro ri'):
        FP.select_train(d, 150)
    d['feature'], d['limits_cfg'] = 'base', 'RE2'
    with pytest.raises(ValueError, match='tran CPU'):
        FP.select_train(d, 150)
    assert len(FP.select_train(d, 150, allow_limits=True)) > 0


def test_anchors_consistent_with_harness():
    import load_sweep_collect as L
    for f, meta in L.FEATURES.items():
        assert FP.FEATURE_ARCHETYPE[f] == meta['archetype']
        assert FP.SOCKSHOP_CALL_CHAINS[meta['archetype']]['expected_delta_pct'] == meta['anchor']


def test_extrapolation_flag(pred):
    assert not pred.verdict(100)['extrapolating']
    assert pred.verdict(400)['extrapolating']
