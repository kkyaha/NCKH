# -*- coding: utf-8 -*-
"""DOI UOC LUONG THANG DO NGAY TRONG HAM BARO THAT -- khong cai lai gi.

Ban truoc toi CAI LAI quy trinh BARO va no KHONG trung (27,78% so voi 34,81%,
lech 23/270 ca), nen phep so do khong hop le. Ban nay thay `RobustScaler` trong
namespace cua module `RCAEval.e2e.baro` bang mot scaler khac co CUNG giao dien,
roi goi CHINH hamf `baro`. Moi dong con lai y nguyen, nen chenh lech quan sat
duoc la do DUNG mot thu: uoc luong thang do.

  goc      RobustScaler cua sklearn (IQR, thang=1 khi suy bien)   = BARO nhu xuat ban
  iqr_bo   IQR, nhung DANH DAU cot suy bien de bi loai
  mad      1,4826*MAD
  mad_bo   1,4826*MAD, loai cot suy bien
  std      do lech chuan, loai cot suy bien
"""
import os, sys, gc, io, contextlib, importlib, importlib.util, warnings
import numpy as np, pandas as pd
from scipy.stats import binomtest
warnings.filterwarnings('ignore')
_W=os.path.join(os.path.dirname(os.path.abspath(__file__)),'rq6_wrapper_rcaeval.py')
_s=importlib.util.spec_from_file_location('w',_W); w=importlib.util.module_from_spec(_s)
_a=sys.argv; sys.argv=['x']; _s.loader.exec_module(w); sys.argv=_a

import RCAEval.e2e.baro as mbaro
GOC = mbaro.RobustScaler

class Thang:
    """Cung giao dien fit/transform nhu RobustScaler, chi doi cach uoc luong thang."""
    def __init__(self, kieu): self.kieu=kieu
    def fit(self, X):
        x=np.asarray(X,dtype=float).ravel()
        q1,q2,q3=np.percentile(x,[25,50,75])
        if self.kieu.startswith('iqr'): self.tam=q2; s=q3-q1
        elif self.kieu.startswith('mad'): self.tam=q2; s=1.4826*np.median(np.abs(x-q2))
        else: self.tam=x.mean(); s=x.std()
        self.suy_bien = not (s>0)
        # neu KHONG loai: hanh xu y sklearn (thang=1). Neu loai: danh dau bang NaN.
        if self.suy_bien:
            self.thang = np.nan if self.kieu.endswith('_bo') or self.kieu=='std' else 1.0
        else:
            self.thang = s
        return self
    def transform(self, X):
        x=np.asarray(X,dtype=float)
        return (x-self.tam)/self.thang      # NaN -> diem NaN -> tut xuong cuoi khi sort

KIEU=['goc','iqr_bo','mad','mad_bo','std']

def chay(d,t,kieu):
    if kieu=='goc': mbaro.RobustScaler=GOC
    else: mbaro.RobustScaler=(lambda k=kieu: Thang(k))
    err=io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            o=mbaro.baro(d.copy(), inject_time=t, dataset=None)
        if 'failed' in err.getvalue(): return None
    except Exception:
        return None
    finally:
        mbaro.RobustScaler=GOC
    sv=[]
    for c in (o.get('ranks') or []):
        if c=='time': continue
        v=w.col2svc(c)
        if v not in sv: sv.append(v)
    return sv

rows=[]
for he in ('SockShop','OnlineBoutique','TrainTicket'):
    for inj,ft,run,draw,t in w.runs_of(he):
        d=w.lam_sach(draw); del draw
        r=dict(he=he,injected=inj,fault=ft,run=run)
        for k in KIEU:
            xh=chay(d,t,k)
            r[f'{k}_t1']=(bool(xh) and xh[0]==inj) if xh is not None else np.nan
            r[f'{k}_t3']=(bool(xh) and inj in xh[:3]) if xh is not None else np.nan
        rows.append(r); del d; gc.collect()
    print(f'  {he} xong ({len(rows)})',flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(w.RES,'rq6_vathang_baro.csv'),index=False)
print(f'\n{len(D)} ca | goc = BARO nhu xuat ban\n')
print(f'{"thang do":10s} {"AC@1":>8s} {"AC@3":>8s}   {"SS":>7s} {"OB":>7s} {"TT":>7s}   {"McNemar vs goc":>24s}')
for k in KIEU:
    g=D.groupby('he')[f'{k}_t1'].mean()*100
    A=D[f'{k}_t1'].astype(float); B=D['goc_t1'].astype(float)
    ok=A.notna()&B.notna()
    h=int(((A==1)&(B==0)&ok).sum()); kk=int(((A==0)&(B==1)&ok).sum())
    p=binomtest(h,h+kk,0.5).pvalue if h+kk else np.nan
    mc='' if k=='goc' else f'{h:3d} v {kk:3d}  p={p:9.3g}{" ***" if (h+kk and p<0.05) else ""}'
    print(f'{k:10s} {100*D[f"{k}_t1"].mean():7.2f}% {100*D[f"{k}_t3"].mean():7.2f}%   '
          f'{g.get("SockShop",0):6.1f}% {g.get("OnlineBoutique",0):6.1f}% {g.get("TrainTicket",0):6.1f}%   {mc:>24s}')
