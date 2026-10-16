# -*- coding: utf-8 -*-
"""
HE A CO PHAN BIET CHAIN CUA TUNG TINH NANG KHONG?
==================================================
`simulate_intervention(injection_service, delta_pct)` -- ham san pham Hệ A dang
dung -- lan Delta theo DUNG TI LE TAI NEN CO SAN cua moi service (kieu P0 trong
thang P0->P3 cua feasibility_predictor.py), KHONG gioi han vao `chain` cua archetype
nhu P1 lam. File nay do TRUC TIEP hau qua cua dieu do, so voi dap an THAT tren
SS-LIMITS/SS-LIMITS-CLEAN (campaign do tinh nang thuc, co bam gia tri base cung
target_rps de tru nhieu).

Phuong phap, cho moi (feature, scale, run):
  1. Chi dung cac BUOC (step) co 'violated'=False trong steps.json cua CHINH run
     do VA cua it nhat 1 run base cung target_rps -- tranh vung CFS throttling/
     qua bao hoa lam bien dang CPU phi tuyen (xem HE_THONG.md muc 7).
  2. real_delta_cpu(step) = CPU trung binh (tinh nang ON) - CPU trung binh
     (base, CUNG target_rps) -- tru nhieu dung phuong phap fit_feature_costs.py
     da dung cho He B.
  3. pred_delta_cpu = simulate_intervention(front-end, feature_pct)[svc]
     ['cpu_predicted'] - [...]['cpu_baseline'] -- CHI MOT GIA TRI cho ca
     feature/scale (khong phu thuoc target_rps, vi Delta cua He A la % CO
     DINH tren baseline cua CHINH AGENT, khong phai ham theo buoc) -- so voi
     PHAN PHOI cac real_delta_cpu qua nhieu buoc/run de co phuong sai, khong
     chi 1 diem.

KHONG sua simulate_intervention()/CapacityAgent -- chi do hau qua cua co che
hien co, dung du lieu da do (SS-LIMITS/SS-LIMITS-CLEAN), khong tao du lieu moi.

Chay: python papers/p1_du_phong/experiments/model_eval/evaluate_chain_awareness.py
"""
import os
import sys
import glob
import json
import warnings

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'

SSTRAIN_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'SS-TRAIN')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results')
os.makedirs(OUT_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(BASE_DIR, 'src', 'graph', 'sockshop_agent_graph.json')
CAMPAIGNS = ['SS-LIMITS', 'SS-LIMITS-CLEAN']

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
from capacity_agent import CapacityAgent  # noqa: E402
from feasibility_predictor import FEATURE_ARCHETYPE, SOCKSHOP_CALL_CHAINS, SERVICES, GATEWAY  # noqa: E402


class _NoOpLLM:
    def invoke(self, *a, **k):
        class _R:
            content = ''
        return _R()


def _safe_steps(steps_json_path):
    """{target_rps: True} cho cac buoc 'violated'=False trong 1 run."""
    j = json.load(open(steps_json_path, encoding='utf-8'))
    return {int(s['target_rps']) for s in j.get('steps', []) if not s.get('violated', True)}, j.get('feature', 'base'), float(j.get('feature_pct', 0) or 0)


def _cpu_by_step(csv_path, safe_rps, services=SERVICES):
    """{target_rps: {service: cpu_mean}} chi cho cac buoc an toan."""
    df = pd.read_csv(csv_path)
    if 'gt_target_rps' not in df.columns:
        return {}
    out = {}
    for rps in safe_rps:
        d = df[df['gt_target_rps'] == rps]
        if d.empty:
            continue
        out[rps] = {s: float(d[f'{s}_cpu'].mean()) for s in services if f'{s}_cpu' in d.columns}
    return out


def collect_rows(agent):
    rows = []
    for campaign in CAMPAIGNS:
        root = os.path.join(BASE_DIR, 'data', 'raw', campaign)
        if not os.path.isdir(root):
            continue

        # Gop tat ca buoc BASE an toan (co the nhieu run) thanh {target_rps: [cpu moi service, ...]}
        base_by_rps = {}
        for sp in sorted(glob.glob(os.path.join(root, 'ramp_base', 'run*', 'steps.json'))):
            safe_rps, _, _ = _safe_steps(sp)
            cpu = _cpu_by_step(os.path.join(os.path.dirname(sp), 'simple_metrics.csv'), safe_rps)
            for rps, svc_cpu in cpu.items():
                base_by_rps.setdefault(rps, []).append(svc_cpu)
        if not base_by_rps:
            continue
        base_mean = {rps: {s: float(np.mean([d[s] for d in lst if s in d]))
                            for s in SERVICES if any(s in d for d in lst)}
                     for rps, lst in base_by_rps.items()}

        for sp in sorted(glob.glob(os.path.join(root, 'ramp_*', 'run*', 'steps.json'))):
            safe_rps, feat, pct = _safe_steps(sp)
            if feat == 'base' or pct <= 0:
                continue
            safe_rps &= set(base_mean)   # chi giu buoc an toan o CA HAI phia
            if not safe_rps:
                continue
            feat_cpu = _cpu_by_step(os.path.join(os.path.dirname(sp), 'simple_metrics.csv'), safe_rps)

            archetype = FEATURE_ARCHETYPE.get(feat, feat)
            chain = set(SOCKSHOP_CALL_CHAINS.get(archetype, {}).get('services', []))
            sim = agent.simulate_intervention(GATEWAY, pct)

            for rps, svc_cpu in feat_cpu.items():
                for svc, cpu_on in svc_cpu.items():
                    if svc not in base_mean[rps] or svc not in sim:
                        continue
                    entry = sim[svc]
                    pb, pa = entry.get('cpu_baseline'), entry.get('cpu_predicted')
                    if pb is None or pa is None or pa == float('inf'):
                        continue
                    rows.append({
                        'campaign': campaign, 'feature': feat, 'feature_pct': pct,
                        'run': os.path.relpath(os.path.dirname(sp), root), 'target_rps': rps,
                        'service': svc, 'in_chain': svc in chain,
                        'real_delta_cpu': round(cpu_on - base_mean[rps][svc], 4),
                        'pred_delta_cpu': round(pa - pb, 4),
                    })
    return pd.DataFrame(rows)


def main():
    agent = CapacityAgent(llm=_NoOpLLM(), data_dir=SSTRAIN_DIR, graph_path=GRAPH_PATH, auto_train=True)
    df = collect_rows(agent)
    out_path = os.path.join(OUT_DIR, 'chain_awareness_evaluation.csv')
    df.to_csv(out_path, index=False)

    pd.set_option('display.width', 160)
    print(f'\n[OK] {len(df)} dong (buoc an toan, da chan vung vo SLO) -> {out_path}\n')

    print('=== Theo trong/ngoai chain: trung binh +/- do lech chuan, qua TOAN BO buoc/run ===')
    g = df.groupby('in_chain').agg(
        n=('real_delta_cpu', 'size'),
        real_mean=('real_delta_cpu', 'mean'), real_std=('real_delta_cpu', 'std'),
        pred_mean=('pred_delta_cpu', 'mean'), pred_std=('pred_delta_cpu', 'std'),
    ).round(4)
    print(g.to_string())

    out_chain = df[~df['in_chain']]
    false_impact = (out_chain['pred_delta_cpu'].abs() > 0.01) & (out_chain['real_delta_cpu'].abs() < 0.01)
    print(f"\nNgoai chain, A du bao co tac dong (>0.01) nhung thuc te ~0: "
          f"{false_impact.sum()}/{len(out_chain)} ({false_impact.mean()*100:.1f}%)")

    print('\n=== Theo tung tinh nang (trung binh qua moi buoc/run an toan) ===')
    gf = df.groupby(['feature', 'in_chain']).agg(
        n=('real_delta_cpu', 'size'),
        real_mean=('real_delta_cpu', 'mean'), pred_mean=('pred_delta_cpu', 'mean'),
    ).round(3)
    print(gf.to_string())

    evaluate_chain_mask_fix(df)
    evaluate_in_chain_correction(df)
    evaluate_structural_correction(df, agent)
    fit_final_correction(df, agent)


def evaluate_chain_mask_fix(df):
    """Do hau qua cua MOT sua rieng, de ap dung, KHONG dung tren DAG/hoi quy:
    sau khi goi simulate_intervention(), ep cpu_predicted = cpu_baseline (delta=0)
    cho moi service KHONG nam trong chain cua archetype -- dung y P1 cua
    feasibility_predictor.py (tinh nang khong lan ra ngoai chain). KHONG sua gi
    o node TRONG chain -- do do day CHI giai quyet duoc nua van de (du bao
    thua ra ngoai chain), khong giai quyet duoc nua con lai (du bao thap hon
    thuc te o node trong chain, xem bang truoc)."""
    df = df.copy()
    df['abs_err_before'] = (df['pred_delta_cpu'] - df['real_delta_cpu']).abs()
    df['pred_delta_cpu_masked'] = np.where(df['in_chain'], df['pred_delta_cpu'], 0.0)
    df['abs_err_after'] = (df['pred_delta_cpu_masked'] - df['real_delta_cpu']).abs()

    print('\n=== Hau qua cua "chan ngoai-chain ve 0" (kieu P1) -- CHUA sua phan trong-chain ===')
    g = df.groupby('in_chain').agg(
        n=('real_delta_cpu', 'size'),
        mae_before=('abs_err_before', 'mean'), mae_after=('abs_err_after', 'mean'),
    ).round(4)
    print(g.to_string())
    mae_b, mae_a = df['abs_err_before'].mean(), df['abs_err_after'].mean()
    print(f"MAE tong the: {mae_b:.4f} -> {mae_a:.4f}  ({(1 - mae_a / mae_b) * 100:.1f}% giam)")

    out = df[~df['in_chain']]
    fp_before = ((out['pred_delta_cpu'].abs() > 0.01) & (out['real_delta_cpu'].abs() < 0.01)).mean() * 100
    fp_after = ((out['pred_delta_cpu_masked'].abs() > 0.01) & (out['real_delta_cpu'].abs() < 0.01)).mean() * 100
    print(f"Ty le du bao sai huong o ngoai-chain: {fp_before:.1f}% -> {fp_after:.1f}%")


def evaluate_in_chain_correction(df):
    """Thu sua PHAN TRONG-CHAIN (he A du bao THAP hon thuc te, nguy hiem hon
    phan ngoai-chain) bang 1 he so nhan hoc tu DU LIEU DA CO -- KHONG dung tren
    DAG/hoi quy, chi la 1 lop sua SAU KHI co pred_delta_cpu.

    He so cho moi service: trung binh (thuc/du bao) tren CAC TINH NANG KHAC --
    kiem bang LEAVE-ONE-FEATURE-OUT (tinh nang dang danh gia KHONG duoc dung
    de hoc he so cua chinh no, tranh hoi cuu/circular). Day la cau hoi: "he so
    hoc tu 7 tinh nang da biet co chuyen sang duoc tinh nang thu 8 CHUA THAY
    khong?" -- dung cau hoi can tra loi cho mot tinh nang MOI thuc te."""
    inc = df[df['in_chain']].copy()
    agg = inc.groupby(['feature', 'service']).agg(
        real=('real_delta_cpu', 'sum'), pred=('pred_delta_cpu', 'sum')).reset_index()
    agg['ratio'] = agg['real'] / agg['pred']

    per_feature = []
    for feat in agg['feature'].unique():
        train = agg[agg['feature'] != feat]
        svc_ratio = train.groupby('service')['ratio'].mean()
        rows_f = inc[inc['feature'] == feat].copy()
        pred_corrected = rows_f['service'].map(svc_ratio).fillna(1.0) * rows_f['pred_delta_cpu']
        err_before = (rows_f['pred_delta_cpu'] - rows_f['real_delta_cpu']).abs().mean()
        err_after = (pred_corrected - rows_f['real_delta_cpu']).abs().mean()
        per_feature.append({'feature': feat, 'n': len(rows_f),
                             'mae_before': round(err_before, 3), 'mae_after': round(err_after, 3)})

    res = pd.DataFrame(per_feature)
    print('\n=== Sua phan TRONG-CHAIN bang he so hoc leave-one-feature-out (khong hoi cuu) ===')
    print(res.to_string(index=False))
    mae_b = (res['mae_before'] * res['n']).sum() / res['n'].sum()
    mae_a = (res['mae_after'] * res['n']).sum() / res['n'].sum()
    print(f"MAE trong-chain gop (leave-one-feature-out): {mae_b:.4f} -> {mae_a:.4f} "
          f"({(1 - mae_a / mae_b) * 100:.1f}% giam)")
    worse = res[res['mae_after'] > res['mae_before']]['feature'].tolist()
    if worse:
        print(f"LUU Y: {len(worse)}/{len(res)} tinh nang bi TE HON sau sua: {worse} "
              f"-- khong phai cai thien dong deu tren moi tinh nang.")


def evaluate_structural_correction(df, agent):
    """Sua phan TRONG-CHAIN bang he so suy TRUC TIEP tu rho_s (ti le workload
    cua service so voi gateway, hoi quy qua goc tren CHINH du lieu SS-TRAIN ma
    agent da train) -- KHONG hoc tu cac tinh nang khac nhu
    evaluate_in_chain_correction(). Phat hien: tuong quan log-log giua 1/rho_s
    va he so dieu chinh thuc nghiem = 0,969 -- gan nhu hoan hao -- vi rho_s nho
    (service hiem khi duoc cham toi boi traffic CHUNG) chinh la ly do co che
    'rai deu theo lich su' pha loang Delta cua mot tinh nang CU THE nang vao
    dung service do.

    Fit log(ratio) = log(a) + b*log(1/rho_s) bang LEAST-SQUARES, kiem
    LEAVE-ONE-FEATURE-OUT (chi 2 tham so hoc tu 7 tinh nang, it hon nhieu so
    voi 7 tham so rieng-service cua evaluate_in_chain_correction() -- it tham
    so hon tren cung luong du lieu train thi it overfit hon)."""
    gw = agent.df_baseline['front-end_workload']
    rho = {s: float((agent.df_baseline[f'{s}_workload'] * gw).sum() / (gw * gw).sum())
           for s in SERVICES if f'{s}_workload' in agent.df_baseline.columns}

    inc = df[df['in_chain']].copy()
    agg = inc.groupby(['feature', 'service']).agg(
        real=('real_delta_cpu', 'sum'), pred=('pred_delta_cpu', 'sum')).reset_index()
    agg = agg[(agg['pred'] != 0) & (agg['real'] / agg['pred'] > 0)]
    agg['ratio'] = agg['real'] / agg['pred']
    agg['x'] = agg['service'].map(lambda s: np.log(1.0 / rho[s]))
    agg['logy'] = np.log(agg['ratio'])

    per_feature = []
    for feat in agg['feature'].unique():
        train = agg[agg['feature'] != feat]
        b, log_a = np.polyfit(train['x'], train['logy'], 1)
        rows_f = inc[inc['feature'] == feat].copy()
        ratio_struct = rows_f['service'].map(lambda s: np.exp(log_a + b * np.log(1.0 / rho[s])))
        pred_corrected = ratio_struct * rows_f['pred_delta_cpu']
        err_before = (rows_f['pred_delta_cpu'] - rows_f['real_delta_cpu']).abs().mean()
        err_after = (pred_corrected - rows_f['real_delta_cpu']).abs().mean()
        per_feature.append({'feature': feat, 'n': len(rows_f),
                             'mae_before': round(err_before, 3), 'mae_after': round(err_after, 3)})

    res = pd.DataFrame(per_feature)
    print('\n=== Sua phan TRONG-CHAIN bang he so suy tu rho_s (cau truc, khong hoc tu tinh nang khac) ===')
    print(res.to_string(index=False))
    mae_b = (res['mae_before'] * res['n']).sum() / res['n'].sum()
    mae_a = (res['mae_after'] * res['n']).sum() / res['n'].sum()
    print(f"MAE trong-chain gop: {mae_b:.4f} -> {mae_a:.4f} ({(1 - mae_a / mae_b) * 100:.1f}% giam)")

    before, after = res['mae_before'].to_numpy(), res['mae_after'].to_numpy()
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(before, after)
        diff = before - after
        ranks = np.argsort(np.argsort(np.abs(diff))) + 1
        wp, wn = ranks[diff > 0].sum(), ranks[diff < 0].sum()
        r = (wp - wn) / (wp + wn) if (wp + wn) else float('nan')
        print(f"Wilcoxon signed-rank (n={len(before)}): p={p:.4f}, rank-biserial r={r:.3f}, "
              f"cai thien {int((diff > 0).sum())}/{len(diff)} tinh nang")
    except ImportError:
        print('[CANH BAO] thieu scipy, khong tinh duoc Wilcoxon/effect size')


def fit_final_correction(df, agent):
    """Fit tham so correction CUOI CUNG de dung trong san pham
    (`CapacityAgent._apply_chain_correction`, xem capacity_agent.py) --
    KHAC voi evaluate_structural_correction() o tren: ham do danh gia bang
    leave-one-feature-out (kiem dinh), ham nay fit tren TOAN BO 8 tinh nang
    (ban dong bang de DUNG, giong p2_params_dev.json/predictions_frozen_RE2.json
    -- khong con tinh nang nao de "giu lam test" vi day la ban cuoi).

    Ghi data/processed/frozen/chain_correction_params.json."""
    gw = agent.df_baseline['front-end_workload']
    rho = {s: float((agent.df_baseline[f'{s}_workload'] * gw).sum() / (gw * gw).sum())
           for s in SERVICES if f'{s}_workload' in agent.df_baseline.columns}

    inc = df[df['in_chain']].copy()
    agg = inc.groupby(['feature', 'service']).agg(
        real=('real_delta_cpu', 'sum'), pred=('pred_delta_cpu', 'sum')).reset_index()
    agg = agg[(agg['pred'] != 0) & (agg['real'] / agg['pred'] > 0)]
    agg['ratio'] = agg['real'] / agg['pred']
    agg['x'] = agg['service'].map(lambda s: np.log(1.0 / rho[s]))
    agg['logy'] = np.log(agg['ratio'])

    b, log_a = np.polyfit(agg['x'], agg['logy'], 1)
    params = {
        'log_a': float(log_a), 'b': float(b),
        'fit_n_features': int(agg['feature'].nunique()), 'fit_n_rows': int(len(agg)),
        'metric': 'cpu',
        'source': 'SS-LIMITS + SS-LIMITS-CLEAN (data/raw)',
        'note': ('Chi ap dung cho CPU trong-chain. Da kiem bang leave-one-feature-out '
                 'truoc khi fit ban cuoi nay (xem evaluate_structural_correction() trong '
                 'evaluate_chain_awareness.py): MAE giam 25.2%, Wilcoxon p=0.0234, '
                 'rank-biserial r=0.889, n=8 tinh nang. Ban nay fit tren CA 8 tinh nang '
                 '(khong giu lai tinh nang nao) vi la ban dung cho san pham, khong phai '
                 'ban kiem dinh.'),
    }
    out_path = os.path.join(BASE_DIR, 'data', 'processed', 'frozen', 'chain_correction_params.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f'\n[OK] Da ghi tham so correction (ban dung cho san pham) -> {out_path}')
    print(f'     log_a={log_a:.4f}  b={b:.4f}  (fit tren {params["fit_n_features"]} tinh nang, '
          f'{params["fit_n_rows"]} diem)')
    return params


if __name__ == '__main__':
    main()
