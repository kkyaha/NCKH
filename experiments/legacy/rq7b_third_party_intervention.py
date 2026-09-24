"""Bien the sach: fault tiem vao service THU BA, ca B va C deu KHONG bi tiem.
Loai bo hieu ung 'fat-hand' (stressor lam nhieu truc tiep bien du bao), chi con
lai confounding thuan. Neu R_C -> R_B van vo duoi can thiep o noi khac, thi day
la confounding, khong phai nhieu truc tiep."""
import os, sys, warnings, itertools
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (sau khi gom thu muc con)
sys.path.insert(0,'/Users/ngkky/NCKH/NCKH/experiments')
from rq7_interventional_validity import _iter_runs, mape, SERVICES, MIN_PRE, MIN_POST

rows=[]
for scenario,target,fault,run_id,df,it in _iter_runs():
    pre,post=df[df['time']<it],df[df['time']>=it]
    if len(pre)<MIN_PRE or len(post)<MIN_POST: continue
    others=[s for s in SERVICES if s!=target]
    for metric in ['cpu','mem']:
        for B,C in itertools.permutations(others,2):     # B=target du bao, C=bien giai thich
            rB,wB,rC=f'{B}_{metric}',f'{B}_workload',f'{C}_{metric}'
            if not all(c in df.columns for c in (rB,wB,rC)): continue
            sp=pre[[rB,wB,rC]].dropna(); so=post[[rB,wB,rC]].dropna()
            if len(sp)<MIN_PRE or len(so)<MIN_POST: continue
            out={}
            for name,xc in (('causal',wB),('assoc_RR',rC)):
                X=sp[[xc]].values
                if np.allclose(X.std(),0): out={}; break
                m=LinearRegression(positive=True).fit(X,sp[rB].values)
                out[name]=(mape(sp[rB].values,m.predict(X)),
                           mape(so[rB].values,m.predict(so[[xc]].values)))
            if not out: continue
            rows.append(dict(scenario=scenario,target=target,fault=fault,metric=metric,
                B=B,C=C,
                causal_pre=out['causal'][0],causal_post=out['causal'][1],
                assoc_pre=out['assoc_RR'][0],assoc_post=out['assoc_RR'][1]))
d=pd.DataFrame(rows)
d['causal_deg']=d.causal_post-d.causal_pre
d['assoc_deg']=d.assoc_post-d.assoc_pre
d.to_csv('/Users/ngkky/NCKH/NCKH/data/processed/scm_results/rq7b_third_party_intervention.csv',index=False)
print(f"n={len(d)} cap (B,C) voi fault o service THU BA\n")
print(f"{'predictor':12s} {'pre':>8s} {'post':>8s} {'degrade':>9s}")
print(f"{'causal W_B':12s} {d.causal_pre.median():8.2f} {d.causal_post.median():8.2f} {d.causal_deg.median():+9.2f}")
print(f"{'assoc R_C':12s} {d.assoc_pre.median():8.2f} {d.assoc_post.median():8.2f} {d.assoc_deg.median():+9.2f}")
dd=(d.causal_deg-d.assoc_deg).dropna()
st,p=stats.wilcoxon(dd)
print(f"\nWilcoxon causal vs assoc: n={len(dd)} median_diff={np.median(dd):+.3f} p={p:.3g}")
bp=(d.assoc_pre<d.causal_pre)&(d.assoc_post>d.causal_post)
print(f"dau hieu confounding (assoc tot hon truoc, te hon sau): {int(bp.sum())}/{len(d)} = {100*bp.mean():.1f}%")
print("\ntheo loai fault (median post MAPE):")
print(d.groupby('fault')[['causal_post','assoc_post']].median().round(2).to_string())
