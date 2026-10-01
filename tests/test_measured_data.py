# -*- coding: utf-8 -*-
"""Chot LOP LAM SACH du lieu tu do (src/scm/measured_data.py).

Vi sao can test nay: that bai nguy hiem nhat cua duong nap du lieu KHONG phai crash ma la
NHIEM BAN IM LANG. `data/raw/SS-LIMITS` chua 836 dong base LAN 1970 dong co tinh nang trong
cung mot goc, va `data/raw/SS-PROSP2/SS-PROSP2` tra ve 4967 dong trong do co ramp_login_x1,
ramp_orderfull_x1... tuc chinh TAP DAP AN cua RQ5. Tro thu muc de lay du lieu train se train
tren dap an ma khong bao loi gi -- va tuyen bo tien cuu mat gia tri ma khong ai thay.

Cac chot duoi day KHONG kiem con so mo hinh (chung se doi), chi kiem cac BAT BIEN cua lop loc.
"""

import os
import sys

import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, 'src', 'scm'))

import measured_data as MD  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MD.RAW_DIR, 'SS-TRAIN')),
    reason='can data/raw/SS-TRAIN (du lieu tu do)')


@pytest.fixture(scope='module')
def aud():
    return MD.audit()


def test_thay_ca_hai_kieu_long_thu_muc(aud):
    """Long 2 cap (SS-TRAIN) va 3 cap (SS-PROSP2) deu phai duoc kham pha.

    Diem mu nay la ly do lop ton tai: data_processor.load_multi_service_data('data/raw/SS-PROSP2')
    tra ve None, va data_contract_check.py (glob '*/*/simple_metrics.csv') chi thay ~39% du lieu.
    """
    assert (aud.batch == '').any(), 'khong thay run nao long 2 cap'
    assert (aud.batch != '').any(), 'khong thay run nao long 3 cap (SS-PROSP2)'
    assert set(aud.root) >= {'SS-TRAIN', 'SS-PROSP2'}


@pytest.mark.parametrize('purpose', ['train', 'baseline'])
def test_khong_co_luu_luong_tinh_nang_trong_tap_hoc(purpose):
    """CHOT QUAN TRONG NHAT: tap hoc co che khong duoc chua mot hang tinh nang nao."""
    df, rep = MD.load(purpose)
    assert len(df), f'{purpose} nap ra tap rong'
    assert set(df['__feature'].unique()) == {'base'}, \
        f"{purpose} lot luu luong tinh nang: {sorted(set(df['__feature'].unique()) - {'base'})}"
    assert not rep[rep.used & (rep.feature != 'base')].shape[0]


def test_tap_eval_toan_bo_la_run_co_tinh_nang():
    df, _ = MD.load('eval')
    assert len(df)
    assert 'base' not in set(df['__feature'].unique())


def test_tap_khoa_bi_loai_mac_dinh():
    """track/review la tap KHOA (data_contract_check.py:38): khong duoc lot vao khi chinh mo hinh."""
    df, rep = MD.load('eval')
    assert not (MD.LOCKED_FEATURES & set(df['__feature'].unique())), \
        f'tap KHOA lot vao eval: {MD.LOCKED_FEATURES & set(df["__feature"].unique())}'
    assert (rep.why == 'locked').any(), 'khong run nao bi danh dau locked -- luat da hong?'
    df2, _ = MD.load('eval', keep_locked=True)
    assert MD.LOCKED_FEATURES & set(df2['__feature'].unique()), 'keep_locked=True khong tra lai tap KHOA'


def test_cau_hinh_tran_invalid_bi_loai():
    """C1/C2 bi limits.json:_status danh dau _INVALID (CFS throttling, ngoai pham vi mo hinh)."""
    for purpose in ('train', 'baseline', 'eval'):
        df, _ = MD.load(purpose)
        assert not (MD.INVALID_LIMITS & set(df['__limits'].unique())), \
            f'{purpose} lot cau hinh tran _INVALID'


def test_train_dung_dung_hop_dong_role_train():
    """role=train (data_contract_check.py:96-103): base + limits=none + chua bao hoa."""
    df, _ = MD.load('train')
    assert set(df['__limits'].unique()) == {'none'}, 'tap train chay duoi tran CPU'


def test_baseline_rong_hon_train_va_noi_dung_dung_cach():
    """baseline noi hop dong (cho phep tran RE2) nen phai RONG hon train, va phai bao gom train."""
    dtr, _ = MD.load('train')
    dba, _ = MD.load('baseline')
    assert len(dba) > len(dtr)
    assert set(dtr['__session'].unique()) <= set(dba['__session'].unique())
    assert set(dba['__limits'].unique()) <= {'none', 'RE2'}


def test_steady_only_bo_dung_hang_warm_va_hang_vi_pham():
    """steady_only=True phai bo that (hop dong: 60% cuoi moi bac), va bo it hon khi tat."""
    on, _ = MD.load('baseline', steady_only=True)
    off, _ = MD.load('baseline', steady_only=False)
    assert len(on) < len(off), 'steady_only=True khong bo hang nao'


def test_moi_run_bi_loai_deu_co_ly_do_tra_cuu_duoc():
    """Khong duoc loai bo im lang: moi `why` phai co trong WHY va co noi dung."""
    _, rep = MD.load('baseline')
    for why in set(rep.loc[~rep.used, 'why']):
        assert why in MD.WHY and MD.WHY[why], f'ly do loai bo khong tra cuu duoc: {why}'
    assert (~rep.used).any(), 'khong loai run nao -- luat loc da hong?'


def test_khong_ghi_gi_vao_du_lieu_tho():
    """Lop nay chi DOC. Chot bang mtime cua mot tep du lieu tho."""
    p = os.path.join(MD.RAW_DIR, 'SS-TRAIN', 'level_100', 'run1', 'simple_metrics.csv')
    before = os.stat(p).st_mtime_ns
    MD.load('baseline')
    assert os.stat(p).st_mtime_ns == before, 'du lieu tho bi sua!'
