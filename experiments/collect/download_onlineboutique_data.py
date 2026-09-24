# -*- coding: utf-8 -*-
"""
DOWNLOAD ONLINE BOUTIQUE TELEMETRY DATA FROM HUGGING FACE
==========================================================
Downloads Online Boutique telemetry (metrics.parquet, logs.parquet,
traces.parquet, inject_time.txt) from phamquiluan/RCAEval on Hugging Face
into data/raw/RE2-OB/ -- cung tier RE2 (metrics+logs+traces) va cung 90-case
Cartesian product (5 service x 6 fault x 3 run) da dung cho RE2-SS/RE2-TT
(xac nhan qua cases.parquet chinh thuc cua RCAEval, khong doan ten scenario).

Khac voi download_trainticket_data.py: tai CA logs.parquet va traces.parquet
(khong chi metrics+inject_time), vi day la lan dau du an dung logs/traces
cua benchmark nay tu dau -- "day du" o day nghia la tron ven 1 tier RE2,
khong phai chi phan da dung truoc do cho SockShop/Train Ticket.

Thu muc dat ten RE2-OB (giu nguyen quy uoc goc RCAEval, giong RE2-SS) --
KHONG doi ten data/raw/RE2-SS hay data/raw/trainticket da co, tranh pha
~30 file experiments/*.py dang tro cung.
"""

import os
import sys
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'RE2-OB')
HF_BASE_URL = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"

# Xac nhan tu cases.parquet chinh thuc (735 case, cot 'dataset'=='RE2-OB'),
# khong doan tu quy uoc dat ten -- dung 5 service bi inject fault that.
SERVICES = [
    'checkoutservice',
    'currencyservice',
    'emailservice',
    'productcatalogservice',
    'recommendationservice',
]
FAULTS = ['cpu', 'delay', 'disk', 'loss', 'mem', 'socket']
FILES = ['metrics.parquet', 'logs.parquet', 'traces.parquet', 'inject_time.txt']


def get_scenario_list(all_runs: bool = True):
    scenarios = []
    runs = [1, 2, 3] if all_runs else [1]
    for svc in SERVICES:
        for fault in FAULTS:
            for run_id in runs:
                scenarios.append(f"re2ob_{svc}_{fault}_{run_id}")
    return scenarios


def download_file(url: str, dest_path: str, max_retries: int = 3) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, 'wb') as out_file:
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
    oks = [download_file(f"{HF_BASE_URL}/{scenario_name}/{fname}",
                          os.path.join(dest_dir, fname))
           for fname in FILES]
    return scenario_name, all(oks)


def main():
    parser = argparse.ArgumentParser(description="Download Online Boutique data (RE2-OB) from Hugging Face")
    parser.add_argument("--all", action="store_true", default=True,
                         help="Download all 90 scenarios (default; kept for CLI symmetry with download_trainticket_data.py)")
    parser.add_argument("--subset", action="store_true",
                         help="Download only run 1 of each (service, fault) pair -- 30 scenarios")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent workers")
    args = parser.parse_args()

    scenarios = get_scenario_list(all_runs=not args.subset)
    print("=" * 70)
    print(f"  DOWNLOADING ONLINE BOUTIQUE DATASET (RCAEval / re2ob)")
    print(f"  Total scenarios to fetch: {len(scenarios)}")
    print(f"  Files per scenario: {FILES}")
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
