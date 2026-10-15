# -*- coding: utf-8 -*-
"""THANG DU LAN TRUYEN: tru di phan nhieu dong GIAI THICH DUOC boi cac con.

Che do loi da do: khi phuong phap sai, top-1 la TO TIEN cua thu pham 50-55% ca
(ngau nhien 26%). Tuc nhieu dong lan NGUOC len chuoi goi, va argmax bi hut ve
phia caller. Phep day ha nguon tho truoc day chi thu 25% khoang cach.

Luat moi, khong dung nhan, dung HUONG DA BIET cua do thi goi:
    z(s)        = diem nhieu dong cua service s (max tren cac cot cua no)
    thang_du(s) = z(s) - agg_{c thuoc con(s)} z(c)
Service cao ma con CUNG cao -> nan nhan. Cao ma con IM -> nguon.

Bien the: con truc tiep / toan bo hau due; agg = max / mean; tru he so beta.
KIEM CHUAN: z-baseline cua toi phai xap xi nsigma, neu khong thi moi so sanh
deu bi nhieu (dung loi toi da mac khi cai lai BARO).
"""
import os, sys, gc, json, importlib.util, warnings
import numpy as np, pandas as pd, networkx as nx
from scipy.stats import binomtest
warnings.filterwarnings('ignore')
_W=os.path.join(os.path.dirname(os.path.abspath(__file__)),'rq6_wrapper_rcaeval.py')
_s=importlib.util.spec_from_file_location('w',_W); w=importlib.util.module_from_spec(_s)
_a=sys.argv; sys.argv=['x']; _s.loader.exec_module(w); sys.argv=_a
G={'SockShop':'sockshop_agent_graph.json','OnlineBoutique':'onlineboutique_agent_graph.json',
   'TrainTicket':'trainticket_agent_graph.json'}
def do_thi(he):
    j=json.load(open(os.path.join('src','graph',G[he]),encoding='utf-8'))
    s={n['id'] if isinstance(n,dict) else n for n in j['nodes']}
    g=nx.DiGraph(); g.add_nodes_from(s)
    g.add_edges_from([(e.get('source',e.get('from')),e.get('target',e.get('to'))) for e in j['edges']])
    return g

def diem_service(d,t):
    """z(s) = max tren cac cot cua s cua |dich chuyen chuan hoa|. Bo cot thang do suy bien."""
    tr=d[d['time']<t]; sa=d[d['time']>=t]
    z={}
    for c in d.columns:
        if c=='time': continue
        x=tr[c].dropna().to_numpy(float); y=sa[c].dropna().to_numpy(float)
        if len(x)<10 or len(y)<10: continue
        s=x.std()
        if s<=0: continue
        v=abs(y.mean()-x.mean())/s
        k=w.col2svc(c); z[k]=max(z.get(k,0.0), float(v))
    return z

def xep(z, g, kieu, beta=1.0):
    if kieu=='goc': sc=dict(z)
    else:
        sc={}
        for s,v in z.items():
            if kieu.startswith('con'):   lien=[c for c in g.successors(s)] if s in g else []
            else:                        lien=list(nx.descendants(g,s)) if s in g else []
            vals=[z[c] for c in lien if c in z]
            if not vals: sc[s]=v; continue
            agg=max(vals) if kieu.endswith('max') else float(np.mean(vals))
            sc[s]=v-beta*agg
    return sorted(sc, key=sc.get, reverse=True)

KIEU=['goc','con_max','con_mean','hau_due_max']
rows=[]
for he in ('SockShop','OnlineBoutique','TrainTicket'):
    g=do_thi(he)
    for inj,ft,run,draw,t in w.runs_of(he):
        d=w.lam_sach(draw); del draw
        z=diem_service(d,t)
        if not z: del d; gc.collect(); continue
        r=dict(he=he,injected=inj,fault=ft,run=run)
        for k in KIEU:
            xh=xep(z,g,k)
            r[f'{k}_t1']=bool(xh) and xh[0]==inj
            r[f'{k}_t3']=bool(xh) and inj in xh[:3]
        rows.append(r); del d; gc.collect()
    print(f'  {he} xong ({len(rows)})',flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(w.RES,'rq6_thang_du_lan_truyen.csv'),index=False)

# KIEM CHUAN: z-baseline co xap xi nsigma khong?
N=pd.concat([pd.read_csv(os.path.join(w.RES,f'rq6_wrapper_rcaeval_v2_{h}.csv')) for h in G],ignore_index=True)
N['fallback_0']=N.fallback_0.fillna('')
N=N[(N.phuong_phap=='nsigma')&(N.fallback_0=='')][['he','injected','fault','run','M0_nhu_xuat_ban']]
M=D.merge(N,on=['he','injected','fault','run'],how='inner')
print(f'\n{len(M)} ca ghep duoc')
print(f'KIEM CHUAN: z-baseline cua toi {100*M.goc_t1.mean():.2f}%  vs  nsigma {100*M.M0_nhu_xuat_ban.mean():.2f}%'
      f'   lech {int((M.goc_t1.astype(bool)!=M.M0_nhu_xuat_ban.astype(bool)).sum())}/{len(M)} ca\n')
print(f'{"luat":14s} {"AC@1":>8s} {"AC@3":>8s}   {"SS":>7s} {"OB":>7s} {"TT":>7s}   {"McNemar vs nsigma":>24s}')
for k in KIEU:
    gg=M.groupby('he')[f'{k}_t1'].mean()*100
    a=M[f'{k}_t1'].astype(int); b=M.M0_nhu_xuat_ban.astype(int)
    h=int(((a==1)&(b==0)).sum()); kk=int(((a==0)&(b==1)).sum())
    p=binomtest(h,h+kk,0.5).pvalue if h+kk else np.nan
    print(f'{k:14s} {100*M[f"{k}_t1"].mean():7.2f}% {100*M[f"{k}_t3"].mean():7.2f}%   '
          f'{gg.get("SockShop",0):6.1f}% {gg.get("OnlineBoutique",0):6.1f}% {gg.get("TrainTicket",0):6.1f}%   '
          f'{h:3d} v {kk:3d} p={p:9.3g}{" ***" if (h+kk and p<0.05) else ""}')
print('\nRIENG TRAINTICKET, vs nsigma:')
T=M[M.he=='TrainTicket']
for k in KIEU:
    a=T[f'{k}_t1'].astype(int); b=T.M0_nhu_xuat_ban.astype(int)
    h=int(((a==1)&(b==0)).sum()); kk=int(((a==0)&(b==1)).sum())
    p=binomtest(h,h+kk,0.5).pvalue if h+kk else np.nan
    print(f'  {k:14s} n={len(T):3d}  nsigma {100*b.mean():6.2f}%  luat {100*a.mean():6.2f}%  '
          f'{100*(a.mean()-b.mean()):+6.2f}   {h:3d} v {kk:3d} p={p:8.3g}{" ***" if (h+kk and p<0.05) else ""}')
