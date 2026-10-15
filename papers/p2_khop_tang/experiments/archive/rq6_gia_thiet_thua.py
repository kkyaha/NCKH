# -*- coding: utf-8 -*-
"""KIEM GIA THIET THUA cua mo hinh hang-thap + thua (robust PCA).

Mo hinh: M = L + S, L hang thap (hoat dong binh thuong co tuong quan), S thua (loi).
Phep thu truc tiep, khong can cai RPCA:
  1. chuan hoa cot bang thong ke cua cua so TRUOC khi tiem (bo cot thang do suy bien
     -- chinh phat hien cua ta)
  2. SVD tren cua so TRUOC -> co so khong gian con binh thuong V_k (giu 95% phuong sai)
  3. phan du cua cua so SAU: R = X_sau - X_sau V_k V_k^T
  4. nang luong phan du theo tung cot -> gop ve service (lay max)
  5. do DO TAP TRUNG cua nang luong do: ti le top-1, top-3, so service phu 80%
Neu nhieu dong lan doc duong goi (nhu chung (a) cua ta goi y) thi phan du KHONG thua
va mo hinh chet tu gia thiet.

Thuong: buoc 4 cho luon mot bo xep hang -> AC@1 mien phi.
"""
import os, sys, gc, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
_W=os.path.join(os.path.dirname(os.path.abspath(__file__)),'rq6_wrapper_rcaeval.py')
_s=importlib.util.spec_from_file_location('w',_W); w=importlib.util.module_from_spec(_s)
_a=sys.argv; sys.argv=['x']; _s.loader.exec_module(w); sys.argv=_a

NGUONG=0.95
rows=[]
for he in ('SockShop','OnlineBoutique','TrainTicket'):
    for inj,ft,run,draw,t in w.runs_of(he):
        d=w.lam_sach(draw); del draw
        tr=d[d['time']<t]; sa=d[d['time']>=t]
        cot=[c for c in d.columns if c!='time']
        if len(tr)<20 or len(sa)<20 or len(cot)<5: del d; gc.collect(); continue
        A=tr[cot].to_numpy(dtype=float); B=sa[cot].to_numpy(dtype=float)
        mu=A.mean(0); sd=A.std(0)
        giu=sd>0                                  # bo cot thang do suy bien
        if giu.sum()<5: del d; gc.collect(); continue
        A=(A[:,giu]-mu[giu])/sd[giu]; B=(B[:,giu]-mu[giu])/sd[giu]
        cg=[c for c,k in zip(cot,giu) if k]
        # co so khong gian con binh thuong
        U,S,Vt=np.linalg.svd(A, full_matrices=False)
        ev=S**2; k=int(np.searchsorted(np.cumsum(ev)/ev.sum(), NGUONG)+1)
        k=max(1,min(k,Vt.shape[0]))
        V=Vt[:k].T
        R=B - B@V@V.T                              # phan du sau khi chieu bo khong gian binh thuong
        e=np.sqrt((R**2).sum(0))                   # nang luong tung cot
        # gop ve service
        svc={}
        for c,val in zip(cg,e):
            s=w.col2svc(c); svc[s]=max(svc.get(s,0.0), float(val))
        if not svc: del d; gc.collect(); continue
        v=np.array(sorted(svc.values(), reverse=True)); tong=v.sum()
        xh=sorted(svc, key=svc.get, reverse=True)
        cs=np.cumsum(v)/tong if tong>0 else np.array([1.0])
        rows.append(dict(he=he,injected=inj,fault=ft,run=run,
            n_cot=int(giu.sum()), n_svc=len(svc), k_hang=k,
            ti_le_hang=k/max(1,giu.sum()),
            top1=float(v[0]/tong) if tong>0 else np.nan,
            top3=float(v[:3].sum()/tong) if tong>0 else np.nan,
            svc_phu_80=int(np.searchsorted(cs,0.80)+1),
            ty_le_svc_phu_80=float((np.searchsorted(cs,0.80)+1)/len(svc)),
            dung_t1=(xh[0]==inj), dung_t3=(inj in xh[:3])))
        del d; gc.collect()
    print(f'  {he} xong ({len(rows)})',flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(w.RES,'rq6_gia_thiet_thua.csv'),index=False)
print(f'\n{len(D)} ca\n')
print('=== (1) Cua so BINH THUONG co hang thap that khong? ===')
print(D.groupby('he')[['n_cot','k_hang','ti_le_hang']].median().round(3).to_string())
print()
print('=== (2) Phan du co THUA khong? (gia thiet cot lo cua robust PCA) ===')
print(f'{"he":16s} {"n_svc":>6s} {"top1 share":>11s} {"top3 share":>11s} {"svc phu 80%":>12s} {"= % so service":>15s}')
for he,g in D.groupby('he'):
    print(f'{he:16s} {g.n_svc.median():6.0f} {g.top1.median():10.3f} {g.top3.median():10.3f} '
          f'{g.svc_phu_80.median():11.0f} {100*g.ty_le_svc_phu_80.median():14.1f}%')
print(f'{"TAT CA":16s} {D.n_svc.median():6.0f} {D.top1.median():10.3f} {D.top3.median():10.3f} '
      f'{D.svc_phu_80.median():11.0f} {100*D.ty_le_svc_phu_80.median():14.1f}%')
print()
print('=== (3) Thuong: bo cham "du khong gian con" dung bao nhieu? ===')
print(f'  AC@1 = {100*D.dung_t1.mean():.2f}%   AC@3 = {100*D.dung_t3.mean():.2f}%')
for he,g in D.groupby('he'):
    print(f'    {he:16s} AC@1 {100*g.dung_t1.mean():6.2f}%  AC@3 {100*g.dung_t3.mean():6.2f}%')
