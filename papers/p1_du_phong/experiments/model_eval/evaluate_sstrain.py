# -*- coding: utf-8 -*-
"""
DANH GIA HE A (CapacityAgent) TREN SS-TRAIN -- du lieu tai THAT do nhom tu do
=============================================================================
Hệ A sống (`src/agents/orchestrator.py`) truoc day train tren `RE2-SS` (benchmark
RCAEval gia lap fault-injection). Phat hien qua khao sat: giu nguyen 100% co che
cua A, chi doi nguon du lieu sang SS-TRAIN (tai THAT, khong co fault) lam MAPE CPU
giam manh tren ca 7 service -- xem papers/p1_du_phong/docs/HE_THONG.md / lich su phien lam viec.

File nay LAP LAI dung giao thuc OOD_Gold_Standard (train tai THAP -> test tai CAO)
ma `evaluation_suite.py` dung de ra so RQ1/RQ2 da cong bo, chi doi nguon doc sang
SS-TRAIN -- KHONG sua `data_processor.py`/`evaluation_suite.py` (chi override
`RAW_DATA_DIR` cuc bo trong tien trinh nay).

HAI PHAN, HAI CAU HOI KHAC NHAU:
  A. Fast Path (hoi quy don bien moi service): tai dung nguyen ham
     `run_f1_rmse_benchmark()` cua evaluation_suite.py.
  B. Accurate Path (lan truyen da tang qua do thi): KHONG the chi doi
     RAW_DATA_DIR roi goi lai ham cu, vi CapacityAgent train tren TOAN BO
     thu muc duoc truyen. De giu dung giao thuc train-thap/test-cao ma
     khong sua code san pham, script nay tao mot thu muc TAM (symlink toi
     cac muc level_* THAP) lam data_dir rieng cho ban CapacityAgent dung de
     DANH GIA (tach biet voi instance song trong orchestrator.py), roi goi
     dung `simulate_intervention()` -- ham san pham thuc su dung -- de du
     bao tai cac muc level_* CAO, so voi gia tri DA DO THAT.

Chay: python papers/p1_du_phong/experiments/model_eval/evaluate_sstrain.py
"""
import os
import sys
import tempfile
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

sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'scm'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'agents'))
sys.path.insert(0, os.path.join(BASE_DIR, 'papers', 'p1_du_phong', 'experiments', 'model_eval'))

import data_processor as DP  # noqa: E402
from data_processor import SERVICES  # noqa: E402
import evaluation_suite as ES  # noqa: E402
from capacity_agent import CapacityAgent  # noqa: E402

GATEWAY = 'front-end'
# Cac muc tai co san trong SS-TRAIN, chia 5 muc THAP (train) / 3 muc CAO (test)
# -- gan ty le 67/33 cua giao thuc goc, chan DUNG THEO MUC (khong theo tung
# hang) de khong lap "anh em sinh doi" cung muc tai vao ca train va test.
TRAIN_LEVELS = [10, 25, 50, 75, 100]
TEST_LEVELS = [150, 200, 250]


class _NoOpLLM:
    """CapacityAgent can 1 doi tuong llm de khoi tao, nhung train_fast_path/
    train_accurate_path/simulate_intervention khong goi LLM -- chi dung cho
    assess_capacity() (khong dung o day)."""
    def invoke(self, *a, **k):
        class _R:
            content = ''
        return _R()


# ============================================================
# PHAN A: FAST PATH -- tai dung nguyen run_f1_rmse_benchmark()
# ============================================================
def run_part_a():
    print('=' * 90)
    print('  PHAN A -- FAST PATH (hoi quy don bien), nguon du lieu = SS-TRAIN')
    print('=' * 90)
    DP.RAW_DATA_DIR = SSTRAIN_DIR  # override CUC BO trong tien trinh nay, khong sua file module
    df_results, _ = ES.run_f1_rmse_benchmark()
    df_results.insert(0, 'data_source', 'SS-TRAIN')
    out_path = os.path.join(OUT_DIR, 'test_f1_rmse_evaluation_SSTRAIN.csv')
    df_results.to_csv(out_path, index=False)
    print(f'\n[OK] Phan A -> {out_path} ({len(df_results)} dong)')
    return df_results


# ============================================================
# PHAN B: ACCURATE PATH -- train tren thu muc TAM (chi muc tai THAP),
# danh gia bang dung simulate_intervention() san pham tai cac muc CAO.
# ============================================================
def _load_level_means(level, services=SERVICES):
    """Trung binh workload/cpu moi service tai 1 muc tai, doc truc tiep tu
    simple_metrics.csv (KHONG qua load_multi_service_data vi ham do bo cot
    scenario/level)."""
    level_dir = os.path.join(SSTRAIN_DIR, f'level_{level}')
    frames = []
    for run_id in sorted(os.listdir(level_dir)):
        rp = os.path.join(level_dir, run_id)
        mp, ip = os.path.join(rp, 'simple_metrics.csv'), os.path.join(rp, 'inject_time.txt')
        if not (os.path.isdir(rp) and os.path.exists(mp) and os.path.exists(ip)):
            continue
        with open(ip) as f:
            it = int(f.read().strip())
        df = pd.read_csv(mp)
        tc = 'imte' if 'imte' in df.columns else 'time'
        frames.append(df[df[tc] < it])
    if not frames:
        raise FileNotFoundError(f'khong doc duoc du lieu cho {level_dir}')
    df_all = pd.concat(frames, ignore_index=True)
    out = {}
    for s in services:
        wlc, cc = f'{s}_workload', f'{s}_cpu'
        if wlc in df_all and cc in df_all:
            out[s] = {'workload': float(df_all[wlc].mean()), 'cpu': float(df_all[cc].mean())}
    return out


def run_part_b():
    print('\n' + '=' * 90)
    print('  PHAN B -- ACCURATE PATH (lan truyen da tang qua do thi)')
    print('  Train tren muc tai THAP', TRAIN_LEVELS, '-> du bao muc tai CAO', TEST_LEVELS)
    print('=' * 90)

    # 1) Thu muc TAM chi symlink toi cac muc tai THAP -- CapacityAgent train
    #    tren dung tap nay, khong cham gi toi SS-TRAIN that/instance song.
    #    LUU Y: _detect_metric_file_format() dung os.walk() KHONG theo symlink
    #    thu muc (os.walk mac dinh followlinks=False) -- symlink nguyen thu muc
    #    level_*/run* se khien no khong bao gio thay file ben trong. Phai tao
    #    THU MUC THAT cho level_*/run_*, chi symlink TUNG FILE ben trong (file
    #    symlink van duoc os.walk bao cao binh thuong trong filenames).
    with tempfile.TemporaryDirectory(prefix='sstrain_trainlow_') as tmp:
        for lv in TRAIN_LEVELS:
            src_level = os.path.join(SSTRAIN_DIR, f'level_{lv}')
            dst_level = os.path.join(tmp, f'level_{lv}')
            os.makedirs(dst_level, exist_ok=True)
            for run_id in os.listdir(src_level):
                src_run = os.path.join(src_level, run_id)
                if not os.path.isdir(src_run):
                    continue
                dst_run = os.path.join(dst_level, run_id)
                os.makedirs(dst_run, exist_ok=True)
                for fname in os.listdir(src_run):
                    os.symlink(os.path.join(src_run, fname), os.path.join(dst_run, fname))

        agent = CapacityAgent(llm=_NoOpLLM(), data_dir=tmp, graph_path=GRAPH_PATH, auto_train=True)
        base_gw = float(agent.df_baseline[f'{GATEWAY}_workload'].mean())

        rows = []
        for lv in TEST_LEVELS:
            real = _load_level_means(lv)
            if GATEWAY not in real:
                continue
            delta_pct = (real[GATEWAY]['workload'] / base_gw - 1.0) * 100.0
            sim = agent.simulate_intervention(GATEWAY, delta_pct)
            for svc, truth in real.items():
                pred = sim.get(svc, {}).get('cpu_predicted')
                if pred is None or not np.isfinite(pred) or pred == float('inf'):
                    continue
                err_pct = abs(pred - truth['cpu']) / abs(truth['cpu']) * 100.0 if truth['cpu'] else float('nan')
                rows.append({
                    'data_source': 'SS-TRAIN', 'test_level': lv, 'service': svc,
                    'delta_pct_applied': round(delta_pct, 2),
                    'cpu_actual': round(truth['cpu'], 4), 'cpu_predicted': round(pred, 4),
                    'abs_pct_error': round(err_pct, 2),
                    'cpu_confidence': sim.get(svc, {}).get('cpu_confidence', 'unknown'),
                })

    df = pd.DataFrame(rows)
    summary = (df.groupby('service')['abs_pct_error'].agg(['mean', 'count'])
                 .rename(columns={'mean': 'mape_pct', 'count': 'n_test_points'}).round(2))
    print('\nMAPE full-resolution theo service (toan bo muc tai CAO):')
    print(summary.to_string())

    out_path = os.path.join(OUT_DIR, 'accurate_path_evaluation_SSTRAIN.csv')
    df.to_csv(out_path, index=False)
    print(f'\n[OK] Phan B -> {out_path} ({len(df)} dong)')
    return df


if __name__ == '__main__':
    run_part_a()
    run_part_b()
