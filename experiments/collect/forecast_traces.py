# -*- coding: utf-8 -*-
"""
BO SINH CHUOI TAI CHO BAI TOAN FORECASTING (thuan numpy, khong can Docker)
==========================================================================
Vi sao can: cac lan quet muc tai (load_sweep_collect --levels) cho cac KHOI ON DINH
doc lap -- tot cho "muc tai -> tai nguyen", nhung KHONG phai chuoi thoi gian: khong co
phu thuoc theo thoi gian, xu huong, mua vu hay dot bien, va persistence luon dung
trong mot khoi. Bo sinh nay tao tai bien thien LIEN TUC de co du lieu forecasting dung
nghia. Moi chuoi = mot ke hoach `planned` (req/s tai gateway, ty le tinh nang, co cau
phien) o do phan giai 1 giay; ke hoach cung la BIEN NGOAI SINH DA BIET TRUOC cho bai
toan "du bao tai nguyen khi biet quy dao tai" (what-if / capacity planning).

Ho chuoi (moi ho phuc vu mot kieu de cua forecasting):
  diurnal   mua vu (ngay nen lai thanh chu ky ngan) + nhieu AR(1)
  ramp      xu huong tuyen tinh -> ngoai suy ra ngoai vung train (OOD tail)
  steps     doi che do (level shift) voi thoi gian giu ngau nhien
  randwalk  qua trinh OU tren log(tai): ngau nhien, khong mua vu
  bursts    nen phang + dot bien (flash crowd) suy giam mu
  mixdrift  tong tai khong doi, co cau phien troi: workload tung service tach nhau
  rollout   tinh nang duoc trien khai dan (S-curve) tren nen diurnal
"""

import numpy as np

LO = 10.0                                   # tai san (req/s)
KINDS = ('browse', 'cart', 'order', 'register')
KIND_REQ = (4, 5, 5, 2)                     # so request toi front-end moi phien (khop SESSIONS)
MIXES = {'base': (0.55, 0.25, 0.15, 0.05),
         'browse': (0.85, 0.10, 0.04, 0.01),
         'checkout': (0.25, 0.25, 0.45, 0.05)}
DIURNAL_PERIOD = 300                        # giay: "1 ngay" nen lai con 5 phut


def mean_req(mix):
    return float(np.dot(mix, KIND_REQ))


def _ar1(n, rng, phi=0.95, sigma=0.08):
    """Nhan tinh log-AR(1) co ky vong ~1."""
    e = rng.normal(0.0, sigma * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return np.exp(x - sigma ** 2 / 2)


def _diurnal(t, rng, lo, peak):
    base = lo + (peak - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * t / DIURNAL_PERIOD))   # bat dau o day
    return base * _ar1(len(t), rng)


def make_trace(family, duration, peak, seed, lo=LO, feature_max_scale=2.0):
    """-> dict(rps[T], feat_pct[T], mix[T,4]) o luoi 1 giay. `rps` la tai NEN muc tieu."""
    rng = np.random.default_rng(seed)
    t = np.arange(duration, dtype=float)
    mix = np.tile(MIXES['base'], (duration, 1))
    feat_pct = np.zeros(duration)

    if family == 'diurnal':
        r = _diurnal(t, rng, lo, peak)
    elif family == 'ramp':
        r = (lo + (peak - lo) * t / max(duration - 1, 1)) * _ar1(duration, rng, sigma=0.05)
    elif family == 'steps':
        r, i = np.empty(duration), 0
        levels = np.exp(rng.uniform(np.log(lo), np.log(peak), 64))
        levels[rng.integers(0, 8)] = peak                 # chac chan cham dinh
        k = 0
        while i < duration:
            dwell = int(rng.uniform(45, 120))
            r[i:i + dwell] = levels[k % len(levels)]
            i, k = i + dwell, k + 1
        r = r * _ar1(duration, rng, sigma=0.04)
    elif family == 'randwalk':
        theta, mu = 1 / 120.0, 0.5 * (np.log(lo) + np.log(peak))
        sig = 0.6 * np.sqrt(2 * theta)
        x = np.empty(duration)
        x[0] = mu
        for i in range(1, duration):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sig * rng.normal()
        r = np.exp(x)
    elif family == 'bursts':
        base = max(lo, 0.25 * peak)
        r = base * _ar1(duration, rng, sigma=0.05)
        n_spikes = max(2, int(duration / 80))
        starts = np.sort(rng.uniform(0, duration - 20, n_spikes)).astype(int)
        for j, s0 in enumerate(starts):
            amp = (peak - base) if j == 0 else rng.uniform(0.4, 1.0) * (peak - base)
            tau = rng.uniform(8, 20)
            r[s0:] += amp * np.exp(-(t[s0:] - s0) / tau)
    elif family == 'mixdrift':
        r = 100.0 * _ar1(duration, rng, sigma=0.03)
        third = duration / 3.0
        anchors = [MIXES['base'], MIXES['browse'], MIXES['checkout'], MIXES['base']]
        for i in range(duration):
            seg = min(int(i / third), 2)
            w = (i - seg * third) / third
            mix[i] = (1 - w) * np.array(anchors[seg]) + w * np.array(anchors[seg + 1])
    elif family == 'rollout':
        r = _diurnal(t, rng, lo, peak)
        # ty le nguoi dung dung tinh nang di theo duong S: ~0 -> feature_max_scale x anchor
        feat_pct = feature_max_scale / (1 + np.exp(-(t - duration / 2) / (duration / 12.0)))   # nhan voi anchor sau
    else:
        raise ValueError(f'ho chuoi khong ton tai: {family}')

    # chan CUNG tai dinh khai bao: ranh gioi train/OOD da pre-register khong duoc mo di vi nhieu AR,
    # va peak OOD phai nam trong vung load generator con dang tin (~250 req/s)
    return {'rps': np.clip(r, lo * 0.5, peak), 'feat_scale': feat_pct, 'mix': mix}


def default_plan(duration=720):
    """Ke hoach chia TRUOC (pre-registered). train/val/test_in deu peak <= 150; test_ood peak 250
    (vuot han vung train >= 1.6x, tuong duong giao thuc 'train tai THAP -> test tai CAO' cua repo);
    test_feature = trien khai tung tinh nang moi."""
    P = []

    def add(fam, split, peak, seed, feature=None):
        P.append({'id': f'{fam}_{split}_{seed}', 'family': fam, 'split': split, 'peak': peak,
                  'seed': seed, 'duration': duration, 'feature': feature})
    add('diurnal', 'train', 150, 11)
    add('diurnal', 'val', 150, 12)
    add('diurnal', 'test_in', 150, 13)
    add('diurnal', 'test_ood', 250, 14)
    add('ramp', 'test_ood', 250, 22)
    add('steps', 'train', 150, 31)
    add('steps', 'test_in', 150, 32)
    add('randwalk', 'train', 150, 41)
    add('randwalk', 'val', 150, 42)
    add('bursts', 'train', 150, 51)
    add('bursts', 'test_ood', 250, 52)
    add('mixdrift', 'train', 150, 61)
    for i, f in enumerate(('promo', 'recs', 'track', 'review')):
        add('rollout', 'test_feature', 100, 70 + i, feature=f)
        P[-1]['id'] = f'rollout_{f}'
    return P
