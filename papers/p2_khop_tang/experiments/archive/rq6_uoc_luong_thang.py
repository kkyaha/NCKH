# -*- coding: utf-8 -*-
"""UOC LUONG THANG DO nao la dung cho bo cham RCA?

Ba diem da do duoc xep thanh mot hinh:
  hang (bat bien hoan toan, khong chia)      41,11%   -> nem mat do lon = nem tin hieu
  IQR  (bat bien, nhung SUY BIEN khi IQR=0)  34,81%   -> thang do vo
  std  (khong bat bien, khong suy bien)      77,78%
Nen tinh chat can khong phai BAT BIEN ma la KHONG SUY BIEN. Script so cac uoc
luong thang do, giu NGUYEN moi thu khac (diem = max |x - tam| / thang tren cua
so sau, tam & thang uoc luong tren cua so truoc -- dung so do cua BARO/nsigma):

  std        do lech chuan            (= nsigma)
  iqr        IQR                      (= BARO)
  iqr_san    IQR, san = luong tu do   -> sua suy bien ma GIU tinh robust
  mad        1,4826 * MAD             -> robust, it suy bien hon IQR
  mad_san    MAD co san
  winsor     std sau khi winsor 5%    -> robust theo kieu khac
Luong tu do = khoang cach duong nho nhat giua hai gia tri khac nhau cua cot;
do la do phan giai THAT cua phep do, nen lay no lam san la co nghia vat ly,
khong phai mot hang so tuy y.
"""
import os, sys, gc, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
_W=os.path.join(os.path.dirname(os.path.abspath(__file__)),'rq6_wrapper_rcaeval.py')
_s=importlib.util.spec_from_file_location('w',_W); w=importlib.util.module_from_spec(_s)
_a=sys.argv; sys.argv=['x']; _s.loader.exec_module(w); sys.argv=_a

def luong_tu(x):
    u=np.unique(x)
    return float(np.diff(u).min()) if len(u)>1 else 0.0

def thang(x, kieu):
    if kieu=='std':     return float(x.std()), float(x.mean())
    q1,q2,q3=np.percentile(x,[25,50,75])
    if kieu=='iqr':     return float(q3-q1), float(q2)
    if kieu=='iqr_san': return max(float(q3-q1), luong_tu(x)), float(q2)
    if kieu in ('mad','mad_san'):
        m=1.4826*float(np.median(np.abs(x-q2)))
        return (max(m,luong_tu(x)) if kieu=='mad_san' else m), float(q2)
    if kieu=='winsor':
        lo,hi=np.percentile(x,[5,95]); z=np.clip(x,lo,hi)
        return float(z.std()), float(z.mean())
    raise ValueError(kieu)

KIEU=['std','iqr','iqr_san','mad','mad_san','winsor']

def cham(d, t, kieu):
    tr, sa = d[d['time']<t], d[d['time']>=t]
    diem=[]
    for c in d.columns:
        if c=='time': continue
        x=tr[c].dropna().to_numpy(dtype=float); y=sa[c].dropna().to_numpy(dtype=float)
        if len(x)<10 or len(y)<10: continue
        s,m=thang(x,kieu)
        if s<=0: continue          # thang do suy bien -> BO cot, khong gan diem vo cuc
        diem.append((c, float(np.max(np.abs(y-m))/s)))
    diem.sort(key=lambda z:-z[1])
    svc=[]
    for c,_ in diem:
        v=w.col2svc(c)
        if v not in svc: svc.append(v)
    return svc

rows=[]
for he in ('SockShop','OnlineBoutique','TrainTicket'):
    for inj,ft,run,draw,t in w.runs_of(he):
        d=w.lam_sach(draw); del draw
        r=dict(he=he,injected=inj,fault=ft,run=run)
        for k in KIEU:
            xh=cham(d,t,k)
            r[f'{k}_t1']=bool(xh) and xh[0]==inj
            r[f'{k}_t3']=bool(xh) and inj in xh[:3]
        rows.append(r); del d; gc.collect()
    print(f'  {he} xong ({len(rows)})',flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(w.RES,'rq6_uoc_luong_thang.csv'),index=False)
print(f'\n{len(D)} ca\n')
print(f'{"uoc luong":12s} {"AC@1":>8s} {"AC@3":>8s}   {"SS":>7s} {"OB":>7s} {"TT":>7s}')
for k in KIEU:
    g=D.groupby('he')[f'{k}_t1'].mean()*100
    print(f'{k:12s} {100*D[f"{k}_t1"].mean():7.2f}% {100*D[f"{k}_t3"].mean():7.2f}%   '
          f'{g.get("SockShop",0):6.1f}% {g.get("OnlineBoutique",0):6.1f}% {g.get("TrainTicket",0):6.1f}%')
