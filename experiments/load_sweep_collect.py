# -*- coding: utf-8 -*-
"""
QUET TAI CO KIEM SOAT + THU THAP TELEMETRY (sinh bo du lieu con thieu)
=======================================================================
Van de da do duoc (experiments/elasticity_transfer_loso.py, signal gate):
ca ba he RCAEval deu KHONG co tin hieu workload->CPU (R^2 trung vi 0.015 /
0.043 / 0.042), nen tren chinh cac he tool duoc trien khai, do chinh xac du
bao la KHONG DO DUOC -- negative control khong tach khoi mo hinh that. Ly do
la thiet ke: benchmark RCA co tinh giu tai PHANG o giai doan normal de loi
tiem vao noi bat, dieu nay xung khac truc tiep voi capacity forecasting (can
bien do tai de hoc). Alibaba co tin hieu nhung ten service bi hash nen nua
Parser/NL khong chay duoc.

Script nay sinh ra thu KHONG bo cong khai nao co: ten service THAT + bien do
tai THAT, tren mot he trien khai duoc cuc bo.

Chon Sock Shop lam mac dinh (khong phai Online Boutique) vi:
  * Online Boutique da BO docker-compose (chuyen sang k8s-first) -> can them
    minikube/kind/kubectl.
  * Sock Shop con docker-compose, va `front-end` ghi ACCESS-LOG dang
    "METHOD /path STATUS DURATION ms" -- da kiem chung trong
    src/scm/callchain_from_logs.py -- nen lay duoc workload VA latency ma
    KHONG can dung Prometheus/Jaeger.
  * Cung he voi RE2-SS da co san, nen du lieu moi SO SANH TRUC TIEP duoc voi
    du lieu fault-injection cu (cung ten service, cung topology, cung taxonomy).

DAU RA: mot thu muc moi muc tai, dung DUNG quy uoc cot cua repo
(`<svc>_workload`, `<svc>_cpu`, `<svc>_mem`, `<svc>_latency-50`) de
src/scm/data_processor.load_multi_service_data() doc duoc ma khong can sua gi.

CANH BAO: script nay CHUA CHAY DUOC LAN NAO (may phat trien khong co Docker).
Coi day la ban thao can kiem tung buoc, khong phai code da kiem chung.

Dung:
    python experiments/load_sweep_collect.py --check        # chi kiem moi truong
    python experiments/load_sweep_collect.py --levels 5,10,25,50,100 --hold 600
"""

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_ROOT = os.path.join(BASE_DIR, 'data', 'raw', 'SS-LOADSWEEP')

# Access-log cua gateway: "METHOD /path STATUS DURATION ms" -- CUNG regex voi
# src/scm/callchain_from_logs.py (giu MOT quy uoc duy nhat cho ca repo).
ACCESS_LOG_RE = re.compile(
    r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)\s+(\d{3})\s+([\d.]+)\s*ms',
    re.IGNORECASE)

GATEWAY = 'front-end'
# Ha tang: dem vao CPU/mem nhung khong coi la "service" cua SCM.
INFRA_SUFFIXES = ('-db', 'rabbitmq', 'session-db')


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def check_env() -> bool:
    ok = True
    for tool in ('docker',):
        r = sh(f'which {tool}')
        print(f"  {tool}: {r.stdout.strip() or 'KHONG CO'}")
        ok &= bool(r.stdout.strip())
    if ok:
        r = sh('docker info --format "{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}"')
        print(f"  daemon: {r.stdout.strip() or r.stderr.strip()}")
        ok &= (r.returncode == 0)
    r = sh('docker compose version')
    print(f"  compose: {r.stdout.strip() or 'KHONG CO'}")
    return ok


def running_services() -> list:
    """Ten container dang chay (bo container load generator)."""
    r = sh("docker ps --format '{{.Names}}'")
    names = [n.strip() for n in r.stdout.splitlines() if n.strip()]
    return [n for n in names if 'load' not in n.lower()]


def sample_stats() -> dict:
    """Mot lan lay docker stats -> {service: {'cpu': %, 'mem': bytes}}."""
    r = sh("docker stats --no-stream --format '{{json .}}'")
    out = {}
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        name = d.get('Name', '')
        cpu = float(str(d.get('CPUPerc', '0')).rstrip('%') or 0)
        mem_raw = str(d.get('MemUsage', '0B')).split('/')[0].strip()
        out[name] = {'cpu': cpu, 'mem': _to_bytes(mem_raw)}
    return out


def _to_bytes(s: str) -> float:
    m = re.match(r'([\d.]+)\s*([KMGT]?i?B)', s, re.IGNORECASE)
    if not m:
        return 0.0
    v, unit = float(m.group(1)), m.group(2).upper().replace('I', '')
    return v * {'B': 1, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9, 'TB': 1e12}.get(unit, 1)


def parse_window_logs(services: list, since_s: int) -> tuple:
    """Doc log cua MOI service trong cua so vua qua.

    Tra ve (per_service_lines, gateway_requests) trong do:
      per_service_lines[svc] = so dong log service do phat ra  -> proxy WORKLOAD
      gateway_requests       = list duration_ms tu access-log cua gateway
                                -> RPS va latency THAT (khong phai proxy)
    """
    per_service = Counter()
    durations = []
    for svc in services:
        r = sh(f'docker logs --since {since_s}s {svc} 2>&1', timeout=120)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
        per_service[svc] = len(lines)
        if svc == GATEWAY:
            for l in lines:
                m = ACCESS_LOG_RE.match(l.strip())
                if m:
                    durations.append(float(m.group(4)))
    return per_service, durations


def run_level(level: int, hold_s: int, interval_s: int, loadgen_cmd: str) -> pd.DataFrame:
    """Chay MOT muc tai: khoi dong loadgen, cho am may, roi lay mau dinh ky."""
    print(f"\n=== MUC TAI: {level} client, giu {hold_s}s ===")
    sh('docker rm -f ss-loadgen 2>/dev/null')
    cmd = loadgen_cmd.format(level=level)
    print(f"  loadgen: {cmd}")
    sh(f'docker run -d --name ss-loadgen --network host {cmd}')

    warmup = min(60, hold_s // 5)
    print(f"  am may {warmup}s...")
    time.sleep(warmup)

    svcs = running_services()
    rows, t_end = [], time.time() + hold_s
    while time.time() < t_end:
        t0 = time.time()
        stats = sample_stats()
        per_svc, durations = parse_window_logs(svcs, interval_s)
        row = {'time': int(t0), 'load_level': level}
        for name, st in stats.items():
            row[f'{name}_cpu'] = st['cpu']
            row[f'{name}_mem'] = st['mem']
            # workload proxy = so dong log/giay cua chinh service do
            row[f'{name}_workload'] = per_svc.get(name, 0) / max(interval_s, 1)
        if durations:
            row[f'{GATEWAY}_latency-50'] = float(pd.Series(durations).quantile(0.50))
            row[f'{GATEWAY}_latency-90'] = float(pd.Series(durations).quantile(0.90))
            row[f'{GATEWAY}_rps'] = len(durations) / max(interval_s, 1)
        rows.append(row)
        time.sleep(max(0, interval_s - (time.time() - t0)))

    sh('docker rm -f ss-loadgen 2>/dev/null')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='chi kiem tra moi truong roi thoat')
    ap.add_argument('--levels', default='5,10,25,50,100',
                     help='cac muc tai (so client dong thoi), ngan cach dau phay')
    ap.add_argument('--hold', type=int, default=600, help='giay giu moi muc')
    ap.add_argument('--interval', type=int, default=10, help='giay giua 2 lan lay mau')
    ap.add_argument('--loadgen',
                     default='weaveworksdemos/load-test -h localhost -r 100 -c {level}',
                     help='lenh docker run cho load generator; {level} se duoc thay the')
    args = ap.parse_args()

    print("=== kiem tra moi truong ===")
    ok = check_env()
    if args.check:
        return
    if not ok:
        print("\n[DUNG] Chua co Docker. Cai truoc:\n"
              '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
              "  brew install colima docker docker-compose\n"
              "  colima start --cpu 6 --memory 10 --disk 60\n"
              "  curl -sL https://raw.githubusercontent.com/microservices-demo/microservices-demo/"
              "master/deploy/docker-compose/docker-compose.yml -o docker-compose.yml && docker compose up -d")
        return

    svcs = running_services()
    if not svcs:
        print("[DUNG] Khong co container nao dang chay -- `docker compose up -d` truoc.")
        return
    print(f"  service dang chay ({len(svcs)}): {svcs}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    all_frames = []
    for level in [int(x) for x in args.levels.split(',')]:
        df = run_level(level, args.hold, args.interval, args.loadgen)
        d = os.path.join(OUT_ROOT, f'level_{level}')
        os.makedirs(d, exist_ok=True)
        df.to_csv(os.path.join(d, 'metrics.csv'), index=False)
        # inject_time.txt: quy uoc cua repo -- khong co fault nao duoc tiem nen
        # dat o vo cuc => TOAN BO du lieu duoc coi la giai doan "binh thuong".
        with open(os.path.join(d, 'inject_time.txt'), 'w') as f:
            f.write(str(2 ** 31 - 1))
        print(f"  [OK] {len(df)} mau -> {d}/metrics.csv")
        all_frames.append(df)

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        merged.to_csv(os.path.join(OUT_ROOT, 'all_levels.csv'), index=False)
        print(f"\n[OK] gop {len(merged)} mau -> {OUT_ROOT}/all_levels.csv")
        # Bao cao ngay bien do tai dat duoc -- day la CHINH cai bo du lieu cu
        # thieu, nen phai kiem truoc khi tin bat ky ket qua nao sau do.
        wl = [c for c in merged.columns if c.endswith('_workload')]
        if wl:
            cv = (merged[wl].std() / merged[wl].mean().replace(0, float('nan'))).dropna()
            print(f"  he so bien thien workload: trung vi={cv.median():.3f} "
                  f"(RE2-SS hien tai ~0.18; can CAO HON de co tin hieu)")


if __name__ == '__main__':
    main()
