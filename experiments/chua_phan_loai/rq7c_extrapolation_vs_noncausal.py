"""Phan dinh hai giai thich canh tranh cho viec R_A -> R_B vo duoi can thiep:
  H1 (ngoai suy): stressor day R_A ra ngoai mien huan luyen -> loi la extrapolation
  H2 (phi nhan qua): quan he R_A->R_B khong phai co che nhan qua -> vo du R_A o trong mien
Phan biet: do ti le diem POST co R_A NAM TRONG mien pre. Neu loi van lon tren
rieng cac diem trong mien -> H2. Neu loi tap trung o diem ngoai mien -> H1."""
import os, sys, warnings
warnings.filterwarnings('ignore'); os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0,'/Users/ngkky/NCKH/NCKH/experiments')
from rq7_interventional_validity import _iter_runs, mape, MIN_PRE, MIN_POST

BP=[('orders','carts'),('orders','shipping')]
rows=[]
for scenario,target,fault,run_id,df,it in _iter_runs():
    if target!='orders': continue
    pre,post=df[df['time']<it],df[df['time']>=it]
    if len(pre)<MIN_PRE or len(post)<MIN_POST: continue
    for A,B in BP:
        rA,rB,wB=f'{A}_cpu',f'{B}_cpu',f'{B}_workload'
        if not all(c in df.columns for c in (rA,rB,wB)): continue
        sp=pre[[rA,rB,wB]].dropna(); so=post[[rA,rB,wB]].dropna()
        if len(sp)<MIN_PRE or len(so)<MIN_POST: continue
        lo,hi=sp[rA].min(),sp[rA].max()
        inrange=(so[rA]>=lo)&(so[rA]<=hi)
        m_as=LinearRegression(positive=True).fit(sp[[rA]].values,sp[rB].values)
        m_ca=LinearRegression(positive=True).fit(sp[[wB]].values,sp[rB].values)
        for lab,mask in (('in_range',inrange),('out_of_range',~inrange)):
            s=so[mask]
            if len(s)<20: continue
            rows.append(dict(scenario=scenario,run=run_id,edge=f'{A}->{B}',subset=lab,n=len(s),
                frac=round(100*len(s)/len(so),1),
                assoc_mape=mape(s[rB].values,m_as.predict(s[[rA]].values)),
                causal_mape=mape(s[rB].values,m_ca.predict(s[[wB]].values))))
d=pd.DataFrame(rows)
print("=== Sai so SAU can thiep, tach theo R_A trong/ngoai mien huan luyen ===")
g=d.groupby(['edge','subset']).agg(n_runs=('n','size'),median_frac=('frac','median'),
    assoc=('assoc_mape','median'),causal=('causal_mape','median')).round(2)
print(g.to_string())
print("\nDoc ket qua:")
print("  neu assoc >> causal ngay ca o 'in_range' -> quan he R_A->R_B khong nhan qua (H2)")
print("  neu assoc ~ causal o 'in_range', chi te o 'out_of_range' -> chi la ngoai suy (H1)")
