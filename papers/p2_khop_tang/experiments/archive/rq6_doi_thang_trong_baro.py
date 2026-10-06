# -*- coding: utf-8 -*-
"""DOI DUNG MOT DONG trong BARO: uoc luong thang do. Giu NGUYEN moi thu khac.

Quy trinh goc cua BARO (RCAEval/e2e/baro.py), nguyen van:
    scaler  = RobustScaler().fit(truoc)          # tam = median, thang = IQR
    zscores = scaler.transform(sau)
    diem    = max(zscores)                       # max CO DAU, khong phai |.|
    xep hang giam dan theo diem

sklearn xu ly thang do bang 0 bang `_handle_zeros_in_scale`: dat thang = 1,0.
Do la co che cua loi: cot IQR=0 thuc chat KHONG duoc chuan hoa, nen do lech THO
cua no de moi cot da chuan hoa dung.

Script giu Y NGUYEN quy trinh tren va chi doi `thang`:
    iqr        IQR, thang=1 khi IQR=0        (= BARO goc, de KIEM reimplementation)
    iqr_bo     IQR, BO cot khi IQR=0
    mad        1,4826*MAD, thang=1 khi MAD=0
    mad_bo     1,4826*MAD, BO cot khi MAD=0
    std        do lech chuan, BO cot khi std=0 (de doi chieu)
"""
import os, sys, gc, io, contextlib, importlib.util, warnings
import numpy as np, pandas as pd
from scipy.stats import binomtest
warnings.filterwarnings('ignore')
_W=os.path.join(os.path.dirname(os.path.abspath(__file__)),'rq6_wrapper_rcaeval.py')
_s=importlib.util.spec_from_file_location('w',_W); w=importlib.util.module_from_spec(_s)
_a=sys.argv; sys.argv=['x']; _s.loader.exec_module(w); sys.argv=_a

def thang_cua(x, kieu):
    """Tra (tam, thang, co_bo). Tam luon la median nhu BARO, tru 'std'."""
    q1,q2,q3 = np.percentile(x,[25,50,75])
    if kieu.startswith('iqr'):
        s=float(q3-q1); tam=float(q2)
    elif kieu.startswith('mad'):
        s=1.4826*float(np.median(np.abs(x-q2))); tam=float(q2)
    elif kieu=='std':
        s=float(x.std()); tam=float(x.mean())
    else: raise ValueError(kieu)
    if s<=0:
        if kieu.endswith('_bo') or kieu=='std': return tam, s, True
        return tam, 1.0, False          # y nhu sklearn _handle_zeros_in_scale
    return tam, s, False

def baro_voi(d, t, kieu):
    tr, sa = d[d['time']<t], d[d['time']>=t]
    diem=[]
    for c in d.columns:
        if c=='time': continue
        x=tr[c].dropna().to_numpy(dtype=float); y=sa[c].dropna().to_numpy(dtype=float)
        if len(x)<2 or len(y)<1: continue
        tam,s,bo = thang_cua(x,kieu)
        if bo: continue
        diem.append((c, float(np.max((y-tam)/s))))      # max CO DAU, nhu BARO
    diem.sort(key=lambda z:-z[1])
    svc=[]
    for c,_ in diem:
        v=w.col2svc(c)
        if v not in svc: svc.append(v)
    return svc

KIEU=['iqr','iqr_bo','mad','mad_bo','std']
rows=[]
from RCAEval.e2e.baro import baro as baro_thuc
for he in ('SockShop','OnlineBoutique','TrainTicket'):
    for inj,ft,run,draw,t in w.runs_of(he):
        d=w.lam_sach(draw); del draw
        r=dict(he=he,injected=inj,fault=ft,run=run)
        # BARO THUC, de kiem reimplementation cua toi
        err=io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                o=baro_thuc(d.copy(), inject_time=t, dataset=None)
            if 'failed' in err.getvalue(): raise RuntimeError
            sv=[]
            for c in (o.get('ranks') or []):
                if c=='time': continue
                v=w.col2svc(c)
                if v not in sv: sv.append(v)
            r['THUC_t1']=bool(sv) and sv[0]==inj; r['THUC_t3']=bool(sv) and inj in sv[:3]
        except Exception:
            r['THUC_t1']=np.nan; r['THUC_t3']=np.nan
        for k in KIEU:
            xh=baro_voi(d,t,k)
            r[f'{k}_t1']=bool(xh) and xh[0]==inj
            r[f'{k}_t3']=bool(xh) and inj in xh[:3]
        rows.append(r); del d; gc.collect()
    print(f'  {he} xong ({len(rows)})',flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(w.RES,'rq6_doi_thang_trong_baro.csv'),index=False)
print(f'\n{len(D)} ca\n')
print('KIEM reimplementation: ban cai lai cua toi voi "iqr" phai TRUNG BARO thuc')
kh=int((D.iqr_t1.astype(bool)!=D.THUC_t1.astype(bool)).sum())
print(f'  BARO thuc      AC@1 = {100*D.THUC_t1.mean():.2f}%')
print(f'  "iqr" cua toi  AC@1 = {100*D.iqr_t1.mean():.2f}%     lech {kh}/{len(D)} ca\n')
print(f'{"thang do":10s} {"AC@1":>8s} {"AC@3":>8s}   {"SS":>7s} {"OB":>7s} {"TT":>7s}   {"McNemar vs iqr":>22s}')
for k in KIEU:
    g=D.groupby('he')[f'{k}_t1'].mean()*100
    A=D[f'{k}_t1'].astype(int); B=D['iqr_t1'].astype(int)
    h=int(((A==1)&(B==0)).sum()); kk=int(((A==0)&(B==1)).sum())
    p=binomtest(h,h+kk,0.5).pvalue if h+kk else np.nan
    mc='' if k=='iqr' else f'{h:3d} v {kk:3d}  p={p:8.3g}{" ***" if (h+kk and p<0.05) else ""}'
    print(f'{k:10s} {100*D[f"{k}_t1"].mean():7.2f}% {100*D[f"{k}_t3"].mean():7.2f}%   '
          f'{g.get("SockShop",0):6.1f}% {g.get("OnlineBoutique",0):6.1f}% {g.get("TrainTicket",0):6.1f}%   {mc:>22s}')
