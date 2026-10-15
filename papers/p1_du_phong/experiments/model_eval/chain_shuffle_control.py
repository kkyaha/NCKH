# -*- coding: utf-8 -*-
"""
CONTROL: CHAIN BI XAO -- _apply_chain_correction() co THUC SU dua vao DUNG chain khong?
=========================================================================================
CAU HOI. `_apply_chain_correction()` (capacity_agent.py) lam 2 viec khac nhau:
  (i)  MAT NA nhi phan: service NGOAI chain -> cpu_predicted := baseline (khong doi).
  (ii) He so nhan CHO service TRONG chain: ratio = exp(0.3535 + 0.6714*ln(1/rho_s)) --
       CHI phu thuoc rho_s (hang so RIENG CUA SERVICE, hoc 1 lan tren SS-TRAIN), KHONG
       phu thuoc archetype nao dang duoc hoi.
Da do truoc (evaluate_verdict_level_heA.py): rho_orders=0.034829, rho_payment=0.034829,
rho_shipping=0.034830 -- giong nhau toi 5 chu so. Nghia la (ii) gan CUNG mot he so 13.6x
cho ca 3 service nay BAT KE dang xet archetype nao. Phan DUY NHAT thay doi theo archetype
la MAT NA (i). Vay cai lam kha-thi-gia giam 79.2% -> 6.7% la vi mat na biet DUNG chain,
hay chi vi correction LUON khuech dai tang sau va phan lon 27 cell (qua trong so PLACE_ORDER/
APPLY_PROMO_CODE/RECOMMEND_PRODUCTS) tinh co co orders/payment/shipping trong chain THAT?

TIEN LE TRONG CHINH DU AN: P1_ctrl (He B, chain hoan vi) trung P0 tren 288/288 quyet dinh
-- mot truong hop da do duoc rang "biet chain" tuong la dong gop cau truc hoa ra KHONG
dong gop gi do duoc. File nay chay phep kiem CUNG LOAI cho He A, truoc khi correction duoc
cong bo nhu mot dong gop "chain-aware".

HAI CONTROL (giu nguyen CONG THUC he so theo rho_s -- CHI doi chain duoc truyen vao):
  A_ctrl_allchain   chain := CA 7 service, MOI diem (khong loai ai). Tach rieng vai tro cua
                    MAT NA khoi vai tro cua DO LON he so.
  A_ctrl_permute(s) GIU 'front-end' trong chain cho MOI archetype (no la cong vao -- loai no
                    khong phai mot kich ban doi khang co y nghia vat ly, pha vo gia dinh 1-
                    gateway G0 cua toan he thong, xem papers/p1_du_phong/README.md). HOAN VI
                    (derangement, KHONG diem bat dong) tap con "khong-front-end" cua chain
                    GIUA 10 archetype: archetype i nhan chain cua mot archetype j != i khac.
                    Chay N_PERMUTE=20 hoan vi doc lap (seed 0..19) vi MOT hoan vi co the may
                    hoac khong may -- bao cao trung binh + [min,max] tren 20 lan, khong chi 1.
                    GIOI HAN: kich thuoc chain KHONG duoc giu co dinh qua hoan vi (vd archetype
                    2-service co the nhan chain 7-service va nguoc lai) -- day la phep hoan vi
                    "tho" nhat, giong tinh than P1_ctrl (hoan vi nhan chain, khong giu bat bien
                    gi khac). Khong che dieu nay.

CACH TAI SU DUNG (khong viet lai logic): `ev_heA.phan_A_quyet_dinh_tai_luoi_do(bridge, cell_df)`
chi doc `chain` tu CHINH cell_df duoc truyen vao (qua spec.chain) -- va `_apply_chain_correction`
CHI duoc goi cho 'on' (A1), KHONG anh huong 'off' (A-off khong phu thuoc chain, xac minh = self-
check (a) duoi). Vay thay vi viet lai ham, file nay CHI tao cac BAN SAO cua cell_df voi cot
'chain' bi thay doi, roi goi LAI chinh ham do -- moi dong quyet dinh/thong ke (status, kha_thi_
long/chat, cp95 CI) deu la CUA ev_heA, khong vi't lai cong thuc nao.

SELF-CHECK (bat buoc, chay truoc khi bao cao):
  (a) heA_off_status PHAI GIONG HET nhau giua cell_df_true / cell_df_allchain / moi cell_df_
      permuted (A-off khong doc chain) -- neu khac, co loi dau day, DUNG.
  (b) moi hoan vi phai la derangement THAT (khong archetype nao giu chain cua chinh no) --
      kiem bang assert, khong gia dinh rng.shuffle tu nhien ra derangement.

CHAY LAI:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
      papers/p1_du_phong/experiments/model_eval/chain_shuffle_control.py
RA:
  data/processed/scm_results/chain_shuffle_summary.csv       (A1_true, A_off, A_allchain, + 20 dong permute)
  data/processed/scm_results/chain_shuffle_permute_mapping.csv (archetype -> chain-cua-archetype-nao, moi seed)
  data/processed/scm_results/chain_shuffle_seed0_matrix.csv   (288 dong chi tiet, seed=0, de doi chieu tay)
"""
import contextlib
import io
import os
import random
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = _HERE
while not os.path.isdir(os.path.join(BASE_DIR, 'src')) and BASE_DIR != os.path.dirname(BASE_DIR):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, _HERE)

from capacity_agent import CapacityAgent  # noqa: E402
import evaluate_verdict_level_heA as ev_heA  # noqa: E402

OUT_DIR = ev_heA.OUT_DIR
SS_TRAIN = ev_heA.SS_TRAIN
GRAPH_PATH = ev_heA.GRAPH_PATH
SERVICES = ['front-end', 'catalogue', 'user', 'carts', 'orders', 'payment', 'shipping']
N_PERMUTE = 20


# ======================================================================
# 1. XAY cell_df BIEN THE (CHI doi cot 'chain', moi thu khac giu nguyen)
# ======================================================================
def derangement(n: int, rng: random.Random):
    """Hoan vi cua range(n) KHONG co diem bat dong (derangement). Rejection sampling --
    voi n=10 ty le chap nhan ~1/e ~ 37%, vai lan thu la du, khong can thuat toan rieng."""
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def xay_cell_df_allchain(cell_df_true: pd.DataFrame) -> pd.DataFrame:
    out = cell_df_true.copy()
    out['chain'] = [tuple(sorted(SERVICES))] * len(out)
    return out


def xay_cell_df_permuted(cell_df_true: pd.DataFrame, seed: int):
    """Tra ve (cell_df_moi, mapping) voi mapping = {archetype: archetype_nguon_chain}."""
    archetypes = sorted(cell_df_true['archetype'].unique())
    rng = random.Random(seed)
    perm = derangement(len(archetypes), rng)
    mapping = {archetypes[i]: archetypes[perm[i]] for i in range(len(archetypes))}
    assert all(k != v for k, v in mapping.items()), f"seed={seed}: derangement co diem bat dong -- loi logic"

    non_gateway_by_arch = {
        a: set(cell_df_true.loc[cell_df_true.archetype == a, 'chain'].iloc[0]) - {'front-end'}
        for a in archetypes
    }
    new_chain_by_arch = {a: tuple(sorted(non_gateway_by_arch[mapping[a]] | {'front-end'}))
                          for a in archetypes}

    out = cell_df_true.copy()
    out['chain'] = out['archetype'].map(new_chain_by_arch)
    return out, mapping


# ======================================================================
# 2. CHAY 1 BIEN THE qua CHINH ham cua ev_heA (khong viet lai logic quyet dinh)
# ======================================================================
def chay_1_bien_the(bridge, cell_df_variant, buf):
    with contextlib.redirect_stdout(buf):
        return ev_heA.phan_A_quyet_dinh_tai_luoi_do(bridge, cell_df_variant)


def tom_tat_1_cau_hinh(df: pd.DataFrame, nhan: str, col_status='heA_on_status',
                        col_long='heA_on_kha_thi_long', col_chat='heA_on_kha_thi_chat'):
    """Dung CHINH cong thuc cp95 cua ev_heA/evaluate_verdicts -- khong viet lai."""
    rows = []
    that = df['that_kha_thi'].astype(bool)
    n_kt, n_t = int((~that).sum()), int(that.sum())
    for doc, col in (('long', col_long), ('chat', col_chat)):
        du = df[col].astype(bool)
        fp = int((du & ~that).sum())
        fn = int((~du & that).sum())
        lo_fp, hi_fp = ev_heA._cp95(fp, n_kt)
        lo_fn, hi_fn = ev_heA._cp95(fn, n_t)
        rows.append(dict(cau_hinh=nhan, cach_doc=doc, n=len(df), n_kt=n_kt, n_t=n_t,
                          fp=fp, fp_pct=100 * fp / max(n_kt, 1), fp_ci_lo=lo_fp, fp_ci_hi=hi_fp,
                          fn=fn, fn_pct=100 * fn / max(n_t, 1), fn_ci_lo=lo_fn, fn_ci_hi=hi_fn,
                          dung_pct=100 * float((du == that).mean())))
    return pd.DataFrame(rows)


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 112)
    print("  CONTROL: CHAIN BI XAO -- correction cua He A dua vao DUNG chain, hay chi vao rho_s?")
    print("=" * 112)

    buf = io.StringIO()
    print("\nKhoi tao CapacityAgent (llm=None, SS-TRAIN)...")
    with contextlib.redirect_stdout(buf):
        agent = CapacityAgent(llm=None, data_dir=SS_TRAIN, graph_path=GRAPH_PATH, auto_train=True)
        bridge = ev_heA.HeABridge(agent)
    if not ev_heA._check_evidence_sanity(bridge):
        raise RuntimeError("Kiem tra doi chung evidence (ev_heA) KHONG khop -- dung.")
    print("  [OK] cau bac L->delta khop 10/10 phan quyet doi chung.")

    cell_df_true = ev_heA.nap_27_cell()
    print(f"\n{len(cell_df_true)} cell, {cell_df_true['archetype'].nunique()} archetype.")

    print("\n--- A1 THAT (chain dung) + A-off (tham chieu) ---")
    A_true = chay_1_bien_the(bridge, cell_df_true, buf)
    assert len(A_true) == 288, f"ky vong 288 diem, duoc {len(A_true)}"

    print("--- A_ctrl_allchain (chain = ca 7 service, moi diem) ---")
    cell_df_all = xay_cell_df_allchain(cell_df_true)
    A_all = chay_1_bien_the(bridge, cell_df_all, buf)
    assert len(A_all) == 288

    print(f"--- A_ctrl_permute: {N_PERMUTE} hoan vi doc lap (seed 0..{N_PERMUTE-1}) ---")
    permute_results, mappings = {}, {}
    for seed in range(N_PERMUTE):
        cell_df_p, mapping = xay_cell_df_permuted(cell_df_true, seed)
        permute_results[seed] = chay_1_bien_the(bridge, cell_df_p, buf)
        mappings[seed] = mapping
        print(f"  seed={seed:2d}: {mapping}")

    print("\n--- SELF-CHECK (a): A-off KHONG doi theo chain (giua true/allchain/moi permute) ---")
    off_true = A_true.set_index(['cell', 'muc_tai'])['heA_off_status']
    for nhan, A_x in [('allchain', A_all)] + [(f'permute_seed{s}', permute_results[s]) for s in range(N_PERMUTE)]:
        off_x = A_x.set_index(['cell', 'muc_tai'])['heA_off_status']
        n_lech = int((off_true != off_x.reindex(off_true.index)).sum())
        if n_lech != 0:
            raise RuntimeError(f"[LOI] heA_off_status LECH {n_lech}/288 diem giua true va {nhan} -- "
                                f"co loi dau day (A-off khong duoc phep phu thuoc chain). DUNG.")
    print(f"  [OK] heA_off_status khop 288/288 o ca {1 + N_PERMUTE} bien the -- A-off that su khong doc chain.")

    # ------ Ghi CSV ------
    mapping_rows = [dict(seed=s, archetype=a, chain_vay_tu=src) for s, m in mappings.items() for a, src in m.items()]
    pd.DataFrame(mapping_rows).to_csv(os.path.join(OUT_DIR, 'chain_shuffle_permute_mapping.csv'), index=False)
    permute_results[0].to_csv(os.path.join(OUT_DIR, 'chain_shuffle_seed0_matrix.csv'), index=False)

    # ------ Tong hop ------
    tt_true = tom_tat_1_cau_hinh(A_true, 'A1_true')
    tt_off = tom_tat_1_cau_hinh(A_true, 'A_off', col_status='heA_off_status',
                                 col_long='heA_off_kha_thi_long', col_chat='heA_off_kha_thi_chat')
    tt_all = tom_tat_1_cau_hinh(A_all, 'A_ctrl_allchain')
    tt_permute_each = [tom_tat_1_cau_hinh(permute_results[s], f'A_ctrl_permute_seed{s}') for s in range(N_PERMUTE)]
    tt_permute_cat = pd.concat(tt_permute_each, ignore_index=True)

    # trung binh + [min,max] tren 20 seed, theo (cach_doc)
    agg_rows = []
    for doc in ('long', 'chat'):
        sub = tt_permute_cat[tt_permute_cat.cach_doc == doc]
        agg_rows.append(dict(
            cau_hinh='A_ctrl_permute_MEAN(20 seed)', cach_doc=doc, n=int(sub.n.iloc[0]),
            n_kt=int(sub.n_kt.iloc[0]), n_t=int(sub.n_t.iloc[0]),
            fp=round(sub.fp.mean(), 1), fp_pct=round(sub.fp_pct.mean(), 1),
            fp_min=int(sub.fp.min()), fp_max=int(sub.fp.max()),
            fn=round(sub.fn.mean(), 1), fn_pct=round(sub.fn_pct.mean(), 1),
            fn_min=int(sub.fn.min()), fn_max=int(sub.fn.max()),
            dung_pct=round(sub.dung_pct.mean(), 1),
        ))
    tt_permute_agg = pd.DataFrame(agg_rows)

    tong_hop = pd.concat([tt_true, tt_off, tt_all], ignore_index=True)
    tong_hop.to_csv(os.path.join(OUT_DIR, 'chain_shuffle_summary.csv'), index=False)
    tt_permute_cat.to_csv(os.path.join(OUT_DIR, 'chain_shuffle_permute_each_seed.csv'), index=False)
    print(f"\n[OK] da ghi CSV vao {OUT_DIR}")

    print(f"\n{'='*112}\n  KET QUA: KHA THI GIA (fp) | BO THIEU (fn) -- tren 288 diem\n{'='*112}")
    for doc in ('long', 'chat'):
        print(f"\n  -- cach doc '{doc}' --")
        print(f"   {'cau hinh':28s} {'KHA THI GIA':>22s} {'BO THIEU':>22s}  {'dung%':>7s}")
        for _, r in tong_hop[tong_hop.cach_doc == doc].iterrows():
            print(f"   {r.cau_hinh:28s} {int(r.fp)}/{int(r.n_kt)}={r.fp_pct:5.1f}% [{r.fp_ci_lo:4.1f},{r.fp_ci_hi:4.1f}]%  "
                  f"{int(r.fn)}/{int(r.n_t)}={r.fn_pct:5.1f}% [{r.fn_ci_lo:4.1f},{r.fn_ci_hi:4.1f}]%  {r.dung_pct:6.1f}%")
        r = tt_permute_agg[tt_permute_agg.cach_doc == doc].iloc[0]
        print(f"   {'A_ctrl_permute (TB 20 seed)':28s} {r.fp:5.1f}/{int(r.n_kt)}={r.fp_pct:5.1f}% "
              f"[seed min={int(r.fp_min)},max={int(r.fp_max)}]      "
              f"{r.fn:5.1f}/{int(r.n_t)}={r.fn_pct:5.1f}% [seed min={int(r.fn_min)},max={int(r.fn_max)}]   {r.dung_pct:6.1f}%")

    print(f"\n{'='*112}\n  DOC KET QUA\n{'='*112}")
    fp_true = tong_hop[(tong_hop.cau_hinh == 'A1_true') & (tong_hop.cach_doc == 'long')].fp_pct.iloc[0]
    fp_off = tong_hop[(tong_hop.cau_hinh == 'A_off') & (tong_hop.cach_doc == 'long')].fp_pct.iloc[0]
    fp_all = tong_hop[(tong_hop.cau_hinh == 'A_ctrl_allchain') & (tong_hop.cach_doc == 'long')].fp_pct.iloc[0]
    fp_perm = tt_permute_agg[tt_permute_agg.cach_doc == 'long'].fp_pct.iloc[0]
    print(f"  kha-thi-gia (doc long): A1_true={fp_true:.1f}%  A_off={fp_off:.1f}%  "
          f"A_ctrl_allchain={fp_all:.1f}%  A_ctrl_permute(TB)={fp_perm:.1f}%")
    print("  Neu A_ctrl_* GAN A1_true (xa A_off) -> correction hoat dong du khong biet dung chain")
    print("  (sua lai nhan 'chain-aware' trong bai). Neu A_ctrl_* GAN A_off (xa A1_true) -> mat na")
    print("  chain mang thong tin thuc, correction xung voi ten goi.")
    print(f"\n{'='*112}\n  HET.\n{'='*112}")


if __name__ == '__main__':
    main()
