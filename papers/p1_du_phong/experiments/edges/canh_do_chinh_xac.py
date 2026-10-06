# -*- coding: utf-8 -*-
"""THEM PARENT CO TANG DO CHINH XAC TREN DU LIEU GIU LAI khong?

CAU HOI: them mot parent vao co che `Target ~ own_workload` co that su tang
MAPE / R2 / F1-vuot-nguong tren du lieu HELD-OUT khong -- hay chi lam tang do
khop tren tap train?

Protocol OOD Gold Standard, giong RQ1 / Fast Path: sap xep theo workload cua
CHINH callee, train LOW 67% -> test HIGH 33%. Chia theo workride cua callee chu
khong phai cua caller, de so sanh cong bang giua co va khong co parent moi.

Nguong canh bao: mean + 0.5*std cua `Target` tren TAP TRAIN (khong dung tap test
-- nguong phai co dinh truoc khi xem dap an).

BA CHE DO -- khac nhau DUNG o cot nao la parent goc, cot nao la parent them
---------------------------------------------------------------------------
  canh        : goc = [callee_workload]              them = [caller_cpu]
                mot canh backpressure moi luot, danh gia tung canh rieng le
  da-cha      : goc = [callee_workload]              them = [TAT CA caller_cpu]
                Vi sao can che do nay: `canh` danh gia tung canh doc lap, nhung
                trong do thi THAT da deploy, mot node (vd ts-station-service_cpu)
                co toi 5 caller_cpu lam parent DONG THOI. Che do nay danh gia
                dung mo hinh da trien khai, khong phai tung canh tach roi.
  mem-socket  : goc = [own_workload] + caller_cpu da deploy
                them = [own_mem, own_socket]
                Doi so luong parent, KHONG doi learner -- de cau hoi dung dang
                "them bien nay co giup khong", khong lan sang "doi mo hinh".

Gop tu bon script, nguyen van trong `archive/`:
    archive/backpressure_edge_accuracy_test.py             canh, Sock Shop
    archive/tt_backpressure_edge_accuracy_test.py          canh, Train Ticket
    archive/tt_backpressure_multiparent_accuracy_test.py   da-cha, Train Ticket
    archive/memsocket_edge_accuracy_test.py                mem-socket, Sock Shop
Do trung lap da do: 65% / 51% / 47% / 45% so dong giua cac cap. Ca bon dung cung
phep chia, cung LinearRegression(positive=True), cung quy tac nguong, cung bo chi
so -- nen chung duoc gop quanh mot ham `danh_gia(goc, them)` duy nhat.

Ban gop la tap hop tren: luon tinh ca precision/recall (chi ban Sock Shop `canh`
co), va luon ghi `n_train`. Khong bot chi so nao.

TAI LAP: khong co lay mau Monte-Carlo o day (thuan sklearn), nen tat dinh san;
moi lan `sample()` deu co `random_state=42`. Khac han hai script trong
`dang_co_che_doi_chung.py` / `canh_an_toan_ngoai_suy.py` -- chung dung
`gcm.interventional_samples` nen phai gieo hat.

Chay:
  python papers/p1_du_phong/experiments/edges/canh_do_chinh_xac.py --he sockshop --che-do canh
  python .../canh_do_chinh_xac.py --he trainticket --che-do canh [--nguong-gain 0.03]
  python .../canh_do_chinh_xac.py --he trainticket --che-do da-cha
  python .../canh_do_chinh_xac.py --he sockshop --che-do mem-socket
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_squared_error, precision_score, r2_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES  # noqa: E402

HAT = 42

# Canh backpressure da deploy tren Sock Shop -- ket qua cua replace_vs_add_edge_test.py
# + call_chain_neighbor_diagnostic.py. Dung lam parent GOC o che do mem-socket.
BACKPRESSURE_SS = {'shipping': ['orders'], 'carts': ['orders'], 'user': ['front-end']}
CANH_SS = [('orders', 'shipping'), ('orders', 'carts'), ('front-end', 'user')]

HE = {
    'sockshop': dict(thu_muc=None, chan_doan=None, dich_vu=SERVICES),
    'trainticket': dict(thu_muc=os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'),
                        chan_doan='tt_call_chain_neighbor_diagnostic.csv', dich_vu=None),
}


def _mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


def danh_gia(df, muc_tieu, cot_chia, cot_goc, cot_them):
    """Protocol OOD: sap xep theo `cot_chia` (workload cua callee), train 67% thap
    -> test 33% cao. So mo hinh chi dung `cot_goc` voi mo hinh dung `cot_goc+cot_them`."""
    can = sorted(set(cot_goc + cot_them + [muc_tieu, cot_chia]))
    if not all(c in df.columns for c in can):
        return None
    sub = df[can].dropna()
    if len(sub) < 300:
        return None
    if len(sub) > 4000:
        sub = sub.sample(4000, random_state=HAT)
    sub = sub.sort_values(cot_chia).reset_index(drop=True)
    k = int(len(sub) * 0.67)
    train, test = sub.iloc[:k], sub.iloc[k:]
    if train[cot_chia].nunique() < 3:
        return None

    mo_rong = cot_goc + cot_them
    m_goc = LinearRegression(positive=True).fit(train[cot_goc], train[muc_tieu])
    m_them = LinearRegression(positive=True).fit(train[mo_rong], train[muc_tieu])

    yt = test[muc_tieu].values
    p_goc = m_goc.predict(test[cot_goc]).ravel()
    p_them = m_them.predict(test[mo_rong]).ravel()

    # --- CONG THEO DAI -------------------------------------------------------
    # Co che thai bai cua canh R->R da duoc dinh vi: no sup khi node CHA ra ngoai
    # dai gia tri da thay trong train (do dưới can thiep: suy giam TB 1658 so voi
    # 12,6 cua co che hai tang, n=1080). TRONG dai thi hai dang khong phan biet
    # duoc (ti le suy giam trung vi 1,00x). Nen dat cong dung o CHO VO:
    #   moi hang test, neu MOI cha them con trong [min,max] cua train -> dung m_them
    #   neu khong -> lui ve m_goc
    # Day khong phai mot tham so moi phai fit; nguong lay truc tiep tu tap train.
    # KEP cha them ve bien dai, KHONG chuyen nhanh: chuyen nhanh pha tinh don dieu
    # (do duoc: tut 7,09 tai bien) -> vi pham tien dieu kien (a) cua Menh de 2.
    te_kep = test.copy()
    trong_dai = np.ones(len(test), dtype=bool)
    for c in cot_them:
        lo, hi = float(train[c].min()), float(train[c].max())
        trong_dai &= (test[c].to_numpy() >= lo) & (test[c].to_numpy() <= hi)
        te_kep[c] = np.clip(te_kep[c].to_numpy(), lo, hi)
    p_cong = m_them.predict(te_kep[mo_rong]).ravel()

    du_bao = {'goc': p_goc, 'them': p_them, 'cong_dai': p_cong}

    # Nguong co dinh tu TAP TRAIN -- khong duoc nhin tap test truoc khi dat nguong.
    nguong = train[muc_tieu].mean() + 0.5 * train[muc_tieu].std()
    yt_bin = (yt >= nguong).astype(int)
    suy_bien = yt_bin.sum() in (0, len(yt_bin))     # mot lop duy nhat -> F1 vo nghia

    r = {'n_train': len(train), 'n_test': len(test), 'ty_le_duong': round(float(yt_bin.mean()), 3),
         'n_parent_them': len(cot_them),
         'ty_le_trong_dai': round(float(trong_dai.mean()), 3)}
    for nhan, yp in du_bao.items():
        yp_bin = (yp >= nguong).astype(int)
        r[f'{nhan}_mape'] = round(_mape(yt, yp), 2)
        r[f'{nhan}_rmse'] = round(float(np.sqrt(mean_squared_error(yt, yp))), 4)
        r[f'{nhan}_r2'] = round(r2_score(yt, yp), 4)
        for ten, fn in (('precision', precision_score), ('recall', recall_score), ('f1', f1_score)):
            r[f'{nhan}_{ten}'] = '' if suy_bien else round(fn(yt_bin, yp_bin, zero_division=0), 3)
    return r


def cong_viec(df, he, che_do, nguong_gain):
    """Sinh danh sach (ten hang, muc_tieu, cot_chia, cot_goc, cot_them) theo che do."""
    cf = HE[he]
    if che_do in ('canh', 'da-cha'):
        if cf['chan_doan']:
            d = pd.read_csv(os.path.join(OUT_DIR, cf['chan_doan']))
            manh = d[d['gain'] > nguong_gain]
            theo_callee = manh.groupby('callee')['caller'].apply(list).to_dict()
        else:
            theo_callee = {}
            for caller, callee in CANH_SS:
                theo_callee.setdefault(callee, []).append(caller)
        if che_do == 'da-cha':
            for callee, callers in theo_callee.items():
                yield (dict(callee=callee, n_caller=len(callers), callers=';'.join(callers)),
                       f'{callee}_cpu', f'{callee}_workload',
                       [f'{callee}_workload'], [f'{c}_cpu' for c in callers])
        else:
            for callee, callers in theo_callee.items():
                for caller in callers:
                    yield (dict(caller=caller, callee=callee), f'{callee}_cpu',
                           f'{callee}_workload', [f'{callee}_workload'], [f'{caller}_cpu'])
    elif che_do == 'mem-socket':
        if he != 'sockshop':
            raise SystemExit('che do mem-socket chi dinh nghia cho sockshop '
                             '(BACKPRESSURE_SS la do thi da deploy cua he nay)')
        for s in cf['dich_vu']:
            goc = [f'{s}_workload'] + [f'{c}_cpu' for c in BACKPRESSURE_SS.get(s, [])]
            yield (dict(service=s, co_backpressure=bool(BACKPRESSURE_SS.get(s))),
                   f'{s}_cpu', f'{s}_workload', goc, [f'{s}_mem', f'{s}_socket'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--he', default='sockshop', choices=tuple(HE))
    ap.add_argument('--che-do', default='canh', choices=('canh', 'da-cha', 'mem-socket'))
    ap.add_argument('--nguong-gain', type=float, default=0.03,
                    help='chi dung khi canh duoc doc tu chan doan (trainticket)')
    a = ap.parse_args()

    df = load_multi_service_data(HE[a.he]['thu_muc'], system_type=a.he)
    if a.he == 'sockshop':
        df = df.dropna().reset_index(drop=True)

    rows, bo_qua = [], 0
    for khoa, muc_tieu, cot_chia, goc, them in cong_viec(df, a.he, a.che_do, a.nguong_gain):
        r = danh_gia(df, muc_tieu, cot_chia, goc, them)
        if r is None:
            bo_qua += 1
            continue
        rows.append({**khoa, **r})
        ten = ' -> '.join(str(v) for v in khoa.values())
        print(f"  {ten:<46} MAPE {r['goc_mape']:>7.2f}% -> {r['them_mape']:>7.2f}%   "
              f"R2 {r['goc_r2']:>7.3f} -> {r['them_r2']:>7.3f}")

    if not rows:
        raise SystemExit(f'khong co hang nao danh gia duoc (bo qua {bo_qua})')
    out = pd.DataFrame(rows)
    p = os.path.join(OUT_DIR, f'canh_do_chinh_xac_{a.he}_{a.che_do.replace("-", "_")}.csv')
    out.to_csv(p, index=False)
    print(f'\n[OK] da luu: {p}  (n={len(out)}, bo qua {bo_qua})')

    print('\n' + '=' * 78)
    print(f'  TONG HOP  [{a.he} | {a.che_do}]  n={len(out)}')
    print('=' * 78)
    print(f"  {'chi so':<12} {'goc':>10} {'them parent':>14} {'CONG DAI':>12} {'them>goc':>10} {'cong>goc':>10}")
    for m, tot_hon_la_lon in (('mape', False), ('rmse', False), ('r2', True),
                              ('precision', True), ('recall', True), ('f1', True)):
        g = pd.to_numeric(out[f'goc_{m}'], errors='coerce')
        t = pd.to_numeric(out[f'them_{m}'], errors='coerce')
        c = pd.to_numeric(out[f'cong_dai_{m}'], errors='coerce')
        v = g.notna() & t.notna() & c.notna()
        if not v.any():
            continue
        f = (lambda a, b: (a > b).sum()) if tot_hon_la_lon else (lambda a, b: (a < b).sum())
        print(f'  {m:<12} {g[v].mean():>10.3f} {t[v].mean():>14.3f} {c[v].mean():>12.3f} '
              f'{f"{f(t[v], g[v])}/{v.sum()}":>10} {f"{f(c[v], g[v])}/{v.sum()}":>10}')
    print(f"\n  ti le hang test TRONG dai huan luyen cua cha: "
          f"{100 * out.ty_le_trong_dai.mean():.1f}% (cong chi kich hoat ngoai dai)")


if __name__ == '__main__':
    main()
