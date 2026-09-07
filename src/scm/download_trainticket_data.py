# -*- coding: utf-8 -*-
"""
DOWNLOAD TRAIN TICKET TELEMETRY DATA FROM HUGGING FACE
======================================================
Downloads TrainTicket telemetry (metrics.parquet and inject_time.txt)
from phamquiluan/RCAEval on Hugging Face into data/raw/trainticket/.

Supports downloading 30 representative scenarios (all 5 services x 6 fault types)
or all 90 scenarios with multi-threaded parallel downloads.
"""

import os
import sys
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'trainticket')
HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"

SERVICES = [
    'ts-auth-service',
    'ts-order-service',
    'ts-route-service',
    'ts-train-service',
    'ts-travel-service'
]
FAULTS = ['cpu', 'delay', 'disk', 'loss', 'mem', 'socket']


def get_scenario_list(all_runs: bool = False):
    scenarios = []
    runs = [1, 2, 3] if all_runs else [1]
    for svc in SERVICES:
        for fault in FAULTS:
            for run_id in runs:
                scenarios.append(f"re2tt_{svc}_{fault}_{run_id}")
    return scenarios


def download_file(url: str, dest_path: str, max_retries: int = 3) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"  [ERROR] Failed to download {url}: {e}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False


def download_scenario(scenario_name: str) -> tuple:
    dest_dir = os.path.join(DATA_DIR, scenario_name)
    metrics_url = f"{HF_BASE_URL}/{scenario_name}/metrics.parquet"
    inject_url = f"{HF_BASE_URL}/{scenario_name}/inject_time.txt"
    
    m_ok = download_file(metrics_url, os.path.join(dest_dir, 'metrics.parquet'))
    i_ok = download_file(inject_url, os.path.join(dest_dir, 'inject_time.txt'))
    return scenario_name, m_ok and i_ok


def main():
    parser = argparse.ArgumentParser(description="Download TrainTicket data from Hugging Face")
    parser.add_argument("--all", action="store_true", help="Download all 90 scenarios instead of 30")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent workers")
    args = parser.parse_args()

    scenarios = get_scenario_list(all_runs=args.all)
    print("=" * 70)
    print(f"  DOWNLOADING TRAIN TICKET DATASET (RCAEval / re2tt)")
    print(f"  Total scenarios to fetch: {len(scenarios)}")
    print(f"  Destination directory: {DATA_DIR}")
    print(f"  Concurrency: {args.workers} workers")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_scenario, sc): sc for sc in scenarios}
        for future in as_completed(futures):
            sc_name, ok = future.result()
            if ok:
                success_count += 1
                print(f"  [OK] ({success_count}/{len(scenarios)}) {sc_name}")
            else:
                print(f"  [FAIL] {sc_name}")

    print("\n" + "=" * 70)
    print(f"  DOWNLOAD COMPLETE: {success_count}/{len(scenarios)} scenarios ready.")
    print("=" * 70)


if __name__ == '__main__':
    main()
