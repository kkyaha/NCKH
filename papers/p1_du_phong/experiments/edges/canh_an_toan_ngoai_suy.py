# -*- coding: utf-8 -*-
"""AN TOAN NGOAI SUY khi them canh backpressure vao Global DAG.

CAU HOI: cac canh backpressure moi them vao `capacity_agent.py::train_accurate_path`
co lam TANG so ca sign-inversion -- du bao CPU GIAM khi workload TANG -- o extreme
delta (+150%/+300%) khong, so voi Global DAG cu khong co canh nay?

Vi sao sign-inversion la tieu chi dung: mot mo hinh du phong co the sai so lon ma
van dung duoc, nhung neu no doi DAU thi no bao "an toan" dung luc he dang qua tai.
Day la hong theo huong NGUY HIEM, khac han sai so lon theo huong an toan. Nen phep
kiem nay dem SO CA DOI DAU, khong do MAPE.

Phuong phap: dung lai DUNG cach xay graph cua `train_accurate_path`, chi bat/tat
cac canh backpressure de so sanh CU vs MOI, quet delta tren nhieu injection_service,
kiem dau cua `*_cpu_change_pct` cho moi service.

Gop tu hai script, nguyen van trong `archive/`:
    archive/backpressure_edge_ood_safety_test.py       (Sock Shop)
    archive/tt_backpressure_edge_ood_safety_test.py    (Train Ticket)
Hai ban do trung 65% so dong; khac biet duy nhat la CAU HINH HE (danh sach service,
do thi, dai delta, so mau chieu) va cach lay canh: Sock Shop ghi tay 3 canh, Train
Ticket doc tu `tt_call_chain_neighbor_diagnostic.csv` voi nguong gain. Ca hai duoc
dua vao bang `HE` duoi day, khong con nhan doi logic.

TINH TAI LAP: hai ban goc KHONG gieo hat cho `gcm.interventional_samples`, nen
so dem sign-inversion co the doi giua cac lan chay. Ban nay gieo `HAT = 42


HAT = 42` truoc
moi lan fit va moi lan quet, nen tai lap duoc. So co the khac ban goc -- do la y muon.

Chay:
  python papers/p1_du_phong/experiments/edges/canh_an_toan_ngoai_suy.py --he sockshop
  python .../canh_an_toan_ngoai_suy.py --he trainticket [--nguong-gain 0.03]
"""
import argparse
import json
import os
import sys

import networkx as nx
import pandas as pd
from dowhy import gcm
from dowhy.gcm import AdditiveNoiseModel
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.util.general import set_random_seed
import numpy as np
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
from data_processor import load_multi_service_data, SERVICES, TRAINTICKET_SERVICES  # noqa: E402
from co_che_cong_dai import CoCheCongDai, chi_so_cha_workload  # noqa: E402

HAT = 42


# CoCheCongDai da chuyen vao src/scm/co_che_cong_dai.py (dung chung voi
# capacity_agent.py -- no la code SAN PHAM, khong con la thu nghiem).


HE = {
    'sockshop': dict(
        services=SERVICES,
        do_thi='sockshop_agent_graph.json',
        thu_muc=None,                                  # load_multi_service_data tu tim
        deltas=[5, 20, 50, 100, 150, 300],
        tiem=['front-end', 'orders'],
        n_proj=200,
        # 3 canh ghi tay: ket qua cua replace_vs_add_edge_test.py + call_chain_neighbor_diagnostic.py
        canh=[('orders', 'shipping'), ('orders', 'carts'), ('front-end', 'user')],
        canh_tu=None,
        ra='backpressure_edge_ood_safety.csv',
        tieu_diem=['shipping', 'carts', 'user'],       # noi vua them canh
    ),
    'trainticket': dict(
        services=TRAINTICKET_SERVICES,
        do_thi='trainticket_agent_graph.json',
        thu_muc=os.path.join(BASE_DIR, 'data', 'raw', 'trainticket'),
        deltas=[100, 150, 300],
        tiem=['ts-ui-dashboard', 'ts-preserve-service'],
        n_proj=150,
        canh=None,
        # 50/66 canh vuot nguong gain 0.03 -- khong ghi tay, doc tu chan doan
        canh_tu='tt_call_chain_neighbor_diagnostic.csv',
        ra='tt_backpressure_edge_ood_safety.csv',
        tieu_diem=None,
    ),
}


def lay_canh(cf, nguong):
    """Canh backpressure: ghi tay (Sock Shop) hoac doc tu chan doan + nguong gain (Train Ticket)."""
    if cf['canh'] is not None:
        return cf['canh']
    d = pd.read_csv(os.path.join(OUT_DIR, cf['canh_tu']))
    return list(d[d['gain'] > nguong][['caller', 'callee']].itertuples(index=False, name=None))


def xay_mo_hinh(df_data, cf, canh, cong_dai=False):
    """Dung lai DUNG cach xay graph cua capacity_agent.py::train_accurate_path.
    `canh` rong = phien ban CU (khong co backpressure)."""
    set_random_seed(HAT)
    with open(os.path.join(BASE_DIR, 'src', 'graph', cf['do_thi']), encoding='utf-8') as f:
        gj = json.load(f)
    svc = set(cf['services'])
    g = nx.DiGraph()
    for e in gj.get('edges', []):
        if e['source'] in svc and e['target'] in svc:
            g.add_edge(f"{e['source']}_workload", f"{e['target']}_workload")
    for s in svc:
        for c in (f'{s}_cpu', f'{s}_mem'):             # bo latency: khong lien quan phep kiem nay
            if f'{s}_workload' in df_data.columns and c in df_data.columns:
                g.add_edge(f'{s}_workload', c)
    for caller, callee in canh:
        a, b = f'{caller}_cpu', f'{callee}_cpu'
        if a in df_data.columns and b in df_data.columns:
            g.add_edge(a, b)

    nut = [n for n in g.nodes() if n in df_data.columns]
    g_sub = g.subgraph(nut).copy()
    df_sub = df_data[nut].dropna()
    df_fit = df_sub.sample(min(2000, len(df_sub)), random_state=HAT) if len(df_sub) > 2000 else df_sub

    model = gcm.InvertibleStructuralCausalModel(g_sub)
    gcm.auto.assign_causal_mechanisms(model, df_fit)
    for n in g_sub.nodes():
        if not (n.endswith(('_cpu', '_mem')) or (n.endswith('_workload') and g_sub.in_degree(n) > 0)):
            continue
        an_toan, n_cha = chi_so_cha_workload(g_sub, n)
        # chi gan co che co cong khi node co CA cha workload LAN cha tai nguyen
        if cong_dai and an_toan and len(an_toan) < n_cha:
            uoc = CoCheCongDai(idx_an_toan=an_toan)
        else:
            uoc = LinearRegression(positive=True)
        model.set_causal_mechanism(n, AdditiveNoiseModel(SklearnRegressionModel(uoc)))
    gcm.fit(model, df_fit)
    return model, df_sub


def quet(model, df_sub, cf, tiem, nhan):
    set_random_seed(HAT)
    rows = []
    col_tiem = f'{tiem}_workload'
    if col_tiem not in df_sub.columns:
        return rows
    goc_wl = float(df_sub[col_tiem].mean())
    for delta in cf['deltas']:
        wl = goc_wl * (1 + delta / 100)
        mau = gcm.interventional_samples(
            model, interventions={col_tiem: lambda x, w=wl: w}, num_samples_to_draw=cf['n_proj'])
        for s in cf['services']:
            c = f'{s}_cpu'
            if c not in df_sub.columns or c not in mau.columns:
                continue
            goc_v, du_bao = float(df_sub[c].mean()), float(mau[c].mean())
            chg = (du_bao - goc_v) / abs(goc_v) * 100 if goc_v != 0 else 0.0
            rows.append({'variant': nhan, 'injection_service': tiem, 'delta_pct': delta,
                         'service': s, 'cpu_change_pct': round(chg, 2), 'sign_inverted': chg < 0})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--he', default='sockshop', choices=tuple(HE))
    ap.add_argument('--nguong-gain', type=float, default=0.03,
                    help='chi dung cho he doc canh tu chan doan (trainticket)')
    a = ap.parse_args()
    cf = HE[a.he]

    canh = lay_canh(cf, a.nguong_gain)
    print(f'[{a.he}] so canh backpressure se them: {len(canh)}')
    df_data = load_multi_service_data(cf['thu_muc'], system_type=a.he)

    rows = []
    for c, nhan, cg in [([], 'OLD_no_backpressure', False),
                        (canh, 'NEW_with_backpressure', False),
                        (canh, 'GATED_range_limited', True)]:
        print(f'\n=== Fit {nhan} ===')
        model, df_sub = xay_mo_hinh(df_data, cf, c, cong_dai=cg)
        print(f'  nut={model.graph.number_of_nodes()} canh={model.graph.number_of_edges()}')
        for t in cf['tiem']:
            print(f'  quet injection={t} ...')
            rows += quet(model, df_sub, cf, t, nhan)

    out = pd.DataFrame(rows)
    p = os.path.join(OUT_DIR, cf['ra'])
    out.to_csv(p, index=False)
    print(f'\n[OK] da luu: {p}')

    print('\n' + '=' * 78)
    print('  SO CA SIGN-INVERSION (du bao CPU GIAM khi workload TANG), theo delta_pct')
    print('=' * 78)
    print(out.groupby(['variant', 'delta_pct'])['sign_inverted'].sum().unstack('variant').to_string())

    inv = out[out['sign_inverted']]
    print('\nChi tiet cac ca doi dau:')
    print('  KHONG CO ca nao, trong CA HAI phien ban, tren moi delta/injection/service da quet.'
          if inv.empty else inv.to_string(index=False))

    if cf['tieu_diem']:
        print(f"\nRieng {'/'.join(cf['tieu_diem'])} (noi vua them canh), o delta cao nhat:")
        f = out[out['service'].isin(cf['tieu_diem']) & out['delta_pct'].isin(cf['deltas'][-2:])]
        print(f.pivot_table(index=['injection_service', 'service', 'delta_pct'],
                            columns='variant', values='cpu_change_pct').to_string())


if __name__ == '__main__':
    main()
