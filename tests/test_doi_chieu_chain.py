# -*- coding: utf-8 -*-
"""Kiem thu lop ket hop He A/He B theo chain (src/scm/doi_chieu_chain.py).

CHOT KIEM (xem thao luan trong bao_cao_giao_vien.tex / paper_draft.tex muc
Backpressure Extension va RQ5): neu mot thay doi mai sau lam He A va He B
khong con hoi tu tren 5 node SCORED, test nay FAIL -- bat loi truoc khi no
lan vao mot nhan do tin dang duoc coi la dang tin.
"""
import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'scm'))

import doi_chieu_chain as DCC  # noqa: E402

CSV_OK = os.path.isfile(DCC.DOI_CHIEU_CSV)
pytestmark = pytest.mark.skipif(
    not CSV_OK, reason=f'can {DCC.DOI_CHIEU_CSV} -- chay doi_chieu_hai_he.py truoc')

# Nguong RONG HON con so da do (0,301%) nhung DU CHAT de bat mot thay doi
# thuc: neu He A hoac He B doi co che ma lam hai ben lech xa hon, test nay FAIL.
MAX_LECH_U_NGUONG = 1.0

# 8 archetype dung lam vong tien cuu RQ5 (ANCHOR trong evaluate_frozen.py / FEATURE_ARCHETYPE
# trong feasibility_predictor.py) -- chain lay truc tiep tu SOCKSHOP_CALL_CHAINS, khong viet tay.
CHUOI_8_FEATURE = ['promo', 'recs', 'track', 'review', 'cartsum', 'quickadd', 'express', 'browse']


def test_bang_doi_chieu_ton_tai_va_trong_nguong():
    table = DCC.load_lech_u_table()
    assert set(table) >= DCC.SCORED, 'bang thieu mot trong 5 node SCORED'
    max_scored = max(table[s] for s in DCC.SCORED)
    assert max_scored <= MAX_LECH_U_NGUONG, (
        f'He A/He B lech {max_scored:.3f}% tren node SCORED -- vuot nguong '
        f'{MAX_LECH_U_NGUONG}%, can xem lai gia thiet hoi tu truoc khi dung nhan do tin nay')


def test_front_end_hoi_tu_tuyet_doi():
    """front-end la goc cua CA HAI cong thuc (xem doi_chieu_hai_he.py) -- phai luon 0,00%."""
    table = DCC.load_lech_u_table()
    assert table['front-end'] == 0.0


@pytest.mark.parametrize('services,excepted_excluded', [
    (['front-end', 'orders', 'shipping'], ['shipping']),
    (['front-end', 'user', 'catalogue'], []),
    (['front-end', 'catalogue'], []),
])
def test_cross_check_for_chain_loc_dung_scored(services, excepted_excluded):
    out = DCC.cross_check_for_chain(services)
    assert out['excluded'] == excepted_excluded
    assert out['max_lech_u_pct'] is not None
    assert out['max_lech_u_pct'] <= MAX_LECH_U_NGUONG


def test_cross_check_khong_fit_lai_mo_hinh(monkeypatch):
    """Phai doc tu CSV da tinh san -- KHONG goi lai sklearn/NNLS o request-time."""
    import sklearn.linear_model as lm
    called = []
    orig = lm.LinearRegression.fit
    def _spy(self, *a, **kw):
        called.append(1)
        return orig(self, *a, **kw)
    monkeypatch.setattr(lm.LinearRegression, 'fit', _spy)
    DCC.cross_check_for_chain(['front-end', 'user'])
    assert not called, 'cross_check_for_chain khong duoc fit lai mo hinh o request-time'


def test_8_archetype_vong_tien_cuu_deu_loc_dung():
    """8 feature dung lam headline RQ5 phai tra ve mot can tren hop le, va payment/shipping
    (khi co trong chain) phai bi loai, khong duoc tinh vao nhan do tin."""
    sys.path.insert(0, os.path.join(_PROJ, 'src', 'scm'))
    import feasibility_predictor as FP
    table = DCC.load_lech_u_table()
    for feat in CHUOI_8_FEATURE:
        arch = FP.SOCKSHOP_CALL_CHAINS[FP.FEATURE_ARCHETYPE[feat]]
        out = DCC.cross_check_for_chain(arch['services'], table=table)
        assert out['max_lech_u_pct'] is not None, f'{feat}: khong co service SCORED nao trong chain?'
        assert out['max_lech_u_pct'] <= MAX_LECH_U_NGUONG, f'{feat}: vuot nguong'
        for s in ('payment', 'shipping'):
            if s in arch['services']:
                assert s in out['excluded'], f'{feat}: {s} phai bi loai khoi nhan do tin'
