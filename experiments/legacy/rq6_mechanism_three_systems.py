# -*- coding: utf-8 -*-
"""RQ6 — CO CHE TREN BA HE, MOI NODE CPU (mo rong E4 va E8).

Hai gioi han cu, dong bang script nay:

  (1) E4 (co che tranh chap) chi soi DUNG 4 node, va deu cua Train Ticket.
      Reviewer se hoi: 4 node co dai dien khong? -> day soi MOI node `_cpu`
      co cha, tren CA BA he.

  (2) E8 (quy mo khuech dai) chi co HAI diem tren truc quy mo (7 va 28 service).
      Hai diem thi ve duong gi cung duoc. -> day them Online Boutique (11
      service) thanh BA diem.

Phep do: THANG HIEU DUNG cua tung cha = |coef| * std(cha), voi coef la he so
NNLS (LinearRegression(positive=True)) cua co che DA FIT. Day la phep do TRUC
TIEP tren mo hinh, KHONG qua Shapley -- nen no la mot duong bang chung doc lap
voi moi ket qua quy gan.

  ti phan nhu cau = sum|coef*std| tren cha `_workload`
                    / ( sum|coef*std| tren MOI cha  +  std(phan du) )

PHAI co std(phan du) o mau so. Ban dau toi bo no, va ket qua vo nghia: mot node
chi co DUNG MOT cha (chinh workload cua no) thi ti phan nhu cau = 100% DO XAY
DUNG, bat ke co che do fit giai thich duoc bao nhieu. Ma phan lon node `_cpu`
dung la chi co mot cha, voi R^2 ~ 0 -- tuc co che gan nhu khong giai thich gi.
Khong co phan du trong mau so thi bang so se bao "100% do nhu cau" cho dung
nhung node ma mo hinh khong hieu gi ca.

CANH BAO ve thu tu cha: dowhy.gcm.fit dung sorted(predecessors), KHONG phai thu
tu chen canh cua networkx. Lay sai thu tu thi moi he so bi gan sai nhan ma
khong bao loi. Script giu dung sorted().

Chay:  python experiments/legacy/rq6_mechanism_three_systems.py
Ra:    data/processed/scm_results/rq6_mechanism_three_systems.csv          (1 dong / cap cha-con)
       data/processed/scm_results/rq6_mechanism_three_systems_nodes.csv    (1 dong / node dich)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

_P = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in (os.path.join(_P, 'src'), os.path.join(_P, 'src', 'agents'), os.path.join(_P, 'src', 'scm')):
    sys.path.insert(0, p)
RES = os.path.join(_P, 'data', 'processed', 'scm_results')

from capacity_agent import CapacityAgent  # noqa: E402

HE = [('SockShop', 'sockshop'), ('OnlineBoutique', 'onlineboutique'), ('TrainTicket', 'trainticket')]


def soi_he(ten: str, system_type: str):
    print(f'\n{"=" * 92}\n  {ten}  — dang huan luyen DAG (CapacityAgent)\n{"=" * 92}')
    cap = CapacityAgent(system_type=system_type, auto_train=False)
    cap.train_accurate_path()
    model, df, g = cap.global_dag_model, cap.global_df_baseline, cap.dag_graph

    canh, node = [], []
    muc_tieu = [n for n in g.nodes if n.endswith('_cpu') and g.in_degree(n) > 0]
    print(f'  {g.number_of_nodes()} node, {g.number_of_edges()} canh | '
          f'{len(muc_tieu)} node `_cpu` co cha')

    for tgt in sorted(muc_tieu):
        parents = sorted(g.predecessors(tgt))          # PHAI khop dowhy.gcm.fit
        sk = model.causal_mechanism(tgt).prediction_model.sklearn_model
        if not hasattr(sk, 'coef_'):
            continue
        coefs = np.asarray(sk.coef_).flatten()
        if len(coefs) != len(parents):
            print(f'  [BO QUA] {tgt}: {len(coefs)} he so vs {len(parents)} cha')
            continue
        try:
            r2 = r2_score(df[tgt].values, sk.predict(df[parents].values).flatten())
        except Exception:
            r2 = np.nan

        try:
            resid = float(np.std(df[tgt].values - sk.predict(df[parents].values).flatten()))
        except Exception:
            resid = np.nan

        eff = {}
        for p, c in zip(parents, coefs):
            kind = 'nhu cau' if p.endswith('_workload') else 'tranh chap'
            e = abs(float(c)) * float(df[p].std())
            eff[p] = (kind, e)
            canh.append(dict(he=ten, dich=tgt, cha=p, loai=kind, coef=float(c),
                             std_cha=float(df[p].std()), thang_hieu_dung=e,
                             n_cha=len(parents), r2=r2))
        s_nc = sum(e for k, e in eff.values() if k == 'nhu cau')
        s_tc = sum(e for k, e in eff.values() if k == 'tranh chap')
        tong = s_nc + s_tc + resid                      # PHAN DU o mau so
        node.append(dict(he=ten, dich=tgt, n_cha=len(parents),
                         n_cha_nhu_cau=sum(1 for k, _ in eff.values() if k == 'nhu cau'),
                         n_cha_tranh_chap=sum(1 for k, _ in eff.values() if k == 'tranh chap'),
                         co_tranh_chap=any(k == 'tranh chap' for k, _ in eff.values()),
                         thang_nhu_cau=s_nc, thang_tranh_chap=s_tc, thang_phan_du=resid,
                         ti_phan_nhu_cau=100 * s_nc / tong if tong > 0 else np.nan,
                         ti_phan_tranh_chap=100 * s_tc / tong if tong > 0 else np.nan,
                         ti_phan_phan_du=100 * resid / tong if tong > 0 else np.nan,
                         ti_le_tc_tren_nc=s_tc / s_nc if s_nc > 0 else np.inf,
                         r2=r2, std_dich=float(df[tgt].std())))
    return canh, node


def main():
    canh, node = [], []
    for ten, st in HE:
        try:
            c, n = soi_he(ten, st)
            canh += c
            node += n
        except Exception as e:
            print(f'  [LOI] {ten}: {type(e).__name__}: {e}')

    C, N = pd.DataFrame(canh), pd.DataFrame(node)
    C.to_csv(os.path.join(RES, 'rq6_mechanism_three_systems.csv'), index=False)
    N.to_csv(os.path.join(RES, 'rq6_mechanism_three_systems_nodes.csv'), index=False)

    print(f'\n{"=" * 92}\n  (1) E4 MO RONG — moi node `_cpu`, ba he\n{"=" * 92}')
    with pd.option_context('display.width', 220, 'display.max_rows', 200,
                           'display.float_format', lambda v: f'{v:,.3f}'):
        print(N[['he', 'dich', 'n_cha', 'n_cha_tranh_chap', 'ti_phan_nhu_cau',
                 'ti_phan_tranh_chap', 'ti_phan_phan_du', 'ti_le_tc_tren_nc', 'r2']]
              .to_string(index=False))

    print('\n--- BAO NHIEU node `_cpu` THUC SU co cha tranh chap? ---')
    cc = N.groupby('he').agg(n_node=('dich', 'size'), co_tranh_chap=('co_tranh_chap', 'sum'))
    cc['ti le %'] = (100 * cc.co_tranh_chap / cc.n_node).round(1)
    print(cc.to_string())
    print('\n  -> E4 cu chon DUNG 4 node Train Ticket, va ca 4 DEU co cha tranh chap.')
    print('     Day la MOT MAU DUOC CHON, khong phai mau dai dien. Phai bao ca hai tap.')

    print(f'\n{"=" * 92}\n  (2) E8 MO RONG — BA diem tren truc quy mo\n{"=" * 92}')
    def tong_hop(D):
        return D.groupby('he').agg(
            n_node=('dich', 'size'),
            cha_tb=('n_cha', 'mean'),
            nhu_cau=('ti_phan_nhu_cau', 'median'),
            tranh_chap=('ti_phan_tranh_chap', 'median'),
            phan_du=('ti_phan_phan_du', 'median'),
            r2=('r2', 'median'))
    thu_tu = ['SockShop', 'OnlineBoutique', 'TrainTicket']
    S = tong_hop(N).reindex([h for h in thu_tu if h in N.he.unique()])
    Sc = tong_hop(N[N.co_tranh_chap]).reindex(
        [h for h in thu_tu if h in N[N.co_tranh_chap].he.unique()])
    with pd.option_context('display.width', 220, 'display.float_format', lambda v: f'{v:,.2f}'):
        print('(2a) MOI node `_cpu` co cha:')
        print(S.to_string())
        print('\n(2b) CHI node co cha tranh chap (tap sanh doi voi E4 cu):')
        print(Sc.to_string())

    n_svc = {'SockShop': 7, 'OnlineBoutique': 11, 'TrainTicket': 28}
    if len(S) >= 3:
        x = np.array([n_svc[h] for h in S.index], float)
        y = S.nhu_cau.values
        rho, p = spearmanr(x, y)
        print(f'\n  quy mo (so service) vs ti phan nhu cau:  Spearman rho = {rho:+.3f}  (p = {p:.3f}, n = 3)')
        print('  (n = 3 nen p khong co y nghia — dau va don dieu moi la dieu doc duoc)')
        print(f'  don dieu giam? {"CO" if list(y) == sorted(y, reverse=True) else "KHONG"}')

    print(f'\n  tong: {len(N)} node `_cpu` tren {N.he.nunique()} he '
          f'(truoc day: 4 node tren 1 he)')
    print(f'\n-> {RES}/rq6_mechanism_three_systems.csv  ({len(C)} dong)')
    print(f'-> {RES}/rq6_mechanism_three_systems_nodes.csv  ({len(N)} dong)')


if __name__ == '__main__':
    main()
