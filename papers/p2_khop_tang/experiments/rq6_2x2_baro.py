# -*- coding: utf-8 -*-
"""THIET KE 2x2 SACH tren BARO THAT: (loc cot suy bien) x (uoc luong thang do).

Hai ban truoc deu co loi, da ghi ro de khong ai lap lai:
  (1) cai lai quy trinh BARO -> KHONG trung (27,78% vs 34,81%, lech 23/270 ca)
  (2) danh dau cot suy bien bang NaN -> `sorted()` so sanh NaN luon False nen cot
      NaN roi vao vi tri BAT DINH, khong xuong cuoi. Ket luan "MAD tot nhat" rut
      ra tu ban nay la TAO TAC cua chinh loi do.

Ban nay: BO cot khoi dataframe (sach, khong NaN), va doi scaler bang cach thay
trong namespace module. Luon goi `RCAEval.e2e.baro.baro` that.

  loc   : khong / bo cot co IQR == 0 tren cua so TRUOC khi tiem
  thang : iqr (RobustScaler goc) / std / mad
"""
import os, sys, gc, io, contextlib, warnings, importlib.util
import numpy as np, pandas as pd
from scipy.stats import binomtest
warnings.filterwarnings('ignore')
_W=os.path.join(os.path.dirname(os.path.abspath(__file__)),'rq6_wrapper_rcaeval.py')
_s=importlib.util.spec_from_file_location('w',_W); w=importlib.util.module_from_spec(_s)
_a=sys.argv; sys.argv=['x']; _s.loader.exec_module(w); sys.argv=_a
import RCAEval.e2e.baro as mbaro
GOC=mbaro.RobustScaler

class Thang:
    def __init__(self,kieu): self.kieu=kieu
    def fit(self,X):
        x=np.asarray(X,dtype=float).ravel()
        q1,q2,q3=np.percentile(x,[25,50,75])
        if self.kieu=='mad': self.tam=q2; s=1.4826*np.median(np.abs(x-q2))
        else:                self.tam=x.mean(); s=x.std()
        self.thang = s if s>0 else 1.0        # y nhu sklearn khi suy bien
        return self
    def transform(self,X): return (np.asarray(X,dtype=float)-self.tam)/self.thang

def bo_iqr0(d,t):
    a=d[d['time']<t]; giu=['time']
    for c in d.columns:
        if c=='time': continue
        x=a[c].dropna().to_numpy(dtype=float)
        if len(x)<5: continue
        q1,q3=np.percentile(x,[25,75])
        if q3-q1>0: giu.append(c)
    return d[giu] if len(giu)>3 else d

def chay(d,t,thang):
    mbaro.RobustScaler = GOC if thang=='iqr' else (lambda k=thang: Thang(k))
    err=io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            o=mbaro.baro(d.copy(), inject_time=t, dataset=None)
        if 'failed' in err.getvalue(): return None
    except Exception: return None
    finally: mbaro.RobustScaler=GOC
    sv=[]
    for c in (o.get('ranks') or []):
        if c=='time': continue
        v=w.col2svc(c)
        if v not in sv: sv.append(v)
    return sv

O=[(l,s) for l in ('khong','bo') for s in ('iqr','std','mad')]
rows=[]
for he in ('SockShop','OnlineBoutique','TrainTicket'):
    for inj,ft,run,draw,t in w.runs_of(he):
        d=w.lam_sach(draw); del draw
        dl=bo_iqr0(d,t)
        r=dict(he=he,injected=inj,fault=ft,run=run,
               n_cot=d.shape[1]-1, n_cot_bo=dl.shape[1]-1)
        for l,s in O:
            xh=chay(dl if l=='bo' else d, t, s)
            r[f'{l}_{s}_t1']=(bool(xh) and xh[0]==inj) if xh is not None else np.nan
            r[f'{l}_{s}_t3']=(bool(xh) and inj in xh[:3]) if xh is not None else np.nan
        rows.append(r); del d,dl; gc.collect()
    print(f'  {he} xong ({len(rows)})',flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(w.RES,'rq6_2x2_baro.csv'),index=False)
print(f'\n{len(D)} ca | so cot: {D.n_cot.median():.0f} -> {D.n_cot_bo.median():.0f} sau khi bo\n')
print(f'{"loc":7s} {"thang":6s} {"AC@1":>8s} {"AC@3":>8s}   {"SS":>7s} {"OB":>7s} {"TT":>7s}   {"McNemar vs BARO goc":>26s}')
B=D['khong_iqr_t1'].astype(float)
for l,s in O:
    c=f'{l}_{s}_t1'; A=D[c].astype(float); g=D.groupby('he')[c].mean()*100
    ok=A.notna()&B.notna()
    h=int(((A==1)&(B==0)&ok).sum()); kk=int(((A==0)&(B==1)&ok).sum())
    p=binomtest(h,h+kk,0.5).pvalue if h+kk else np.nan
    mc='  (= BARO goc)' if (l,s)==('khong','iqr') else f'{h:3d} v {kk:3d}  p={p:9.3g}{" ***" if (h+kk and p<0.05) else ""}'
    print(f'{l:7s} {s:6s} {100*D[c].mean():7.2f}% {100*D[f"{l}_{s}_t3"].mean():7.2f}%   '
          f'{g.get("SockShop",0):6.1f}% {g.get("OnlineBoutique",0):6.1f}% {g.get("TrainTicket",0):6.1f}%   {mc:>26s}')
