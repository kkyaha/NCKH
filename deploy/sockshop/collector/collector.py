# -*- coding: utf-8 -*-
"""
COLLECTOR TELEMETRY CHO SOCK SHOP (chay TRONG container, cung mang docker) -- chi dung stdlib
=============================================================================================
Xuat telemetry theo DUNG cau truc nhu RE2-SS/RCAEval (va cac he thong san xuat dung Prometheus/cAdvisor),
de bo du lieu "chuan" so voi telemetry thuong gap; phan ground-truth tinh nang duoc tach rieng o harness
(cot `gt_*`), khong nam trong collector.

Bon dong ra (cung tien to --out, thay `collector_` bang):
  collector_<tag>.csv  DAN XUAT theo cua so (-> simple_metrics.csv): <svc>_{cpu,mem,socket,diskio,workload,
                       error,latency-50/90/95/99/mean}, vm_*
  raw_<tag>.csv        BO DEM TICH LUY kieu cAdvisor/Prometheus, ten cot NHU RE2 (-> metrics.csv):
                       <svc>_container-cpu-usage-seconds-total, -memory-working-set-bytes, -sockets,
                       -spec-cpu-quota (= GIOI HAN CPU, nhu K8s), -network-*, -fs-*, ... + <svc>_app-request-total ...
  logs_<tag>.csv       log container theo schema RE2 logs.csv: time,timestamp(ns),container_name,message,level,req_path,error
                       (giu MOI dong; chi bo ma mau ANSI de dong access-log "METHOD /path STATUS ms" khop nhu RE2)
  routes_<tag>.csv     toc do request theo tung endpoint (method,route,status_class) -- telemetry per-endpoint chuan RED

Cach do (gia tri dan xuat):
  * cpu     : delta bo dem tich luy cgroup / delta t, % cua 1 core (100 = 1 core) -- TRUNG BINH ca cua so
  * mem     : usage - inactive_file (working set), byte
  * socket  : so socket dang mo cua container (/proc/<pid>/net/sockstat; can --pid=host --cap-add SYS_PTRACE)
  * diskio  : (doc + ghi) byte/giay tu blkio
  * workload: delta request_duration_seconds_count / delta t o /metrics THAT cua service (bo route 'metrics')
  * latency : phan vi tu delta bucket histogram (noi suy tuyen tinh nhu histogram_quantile), GIAY
"""

import argparse
import calendar
import csv
import http.client
import json
import math
import re
import socket
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

METRICS_PORT = {'front-end': 8079, 'catalogue': 80, 'user': 80, 'payment': 80,
                'carts': 80, 'orders': 80, 'shipping': 80}
NAN = float('nan')
NCPU = [0]          # so vCPU cua VM (docker /info), dat khi khoi dong

_LINE = re.compile(r'^(request_duration_seconds_(?:count|sum|bucket))\{(.*)\}\s+(\S+)\s*$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_ACCESS = re.compile(r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+\d{3}\s+[\d.]+\s*ms', re.I)
LOG_MSG_MAX, LOG_LINES_PER_TICK = 2000, 30000     # giu MOI dong log nhu RE2-SS (~120 KB/s o 250 req/s)

# ten cot raw (RE2) <- khoa noi bo
RAW_CONT = [
    ('cpu_total', 'container-cpu-usage-seconds-total'), ('cpu_user', 'container-cpu-user-seconds-total'),
    ('cpu_system', 'container-cpu-system-seconds-total'), ('mem_ws', 'container-memory-working-set-bytes'),
    ('mem_usage', 'container-memory-usage-bytes'), ('mem_rss', 'container-memory-rss'),
    ('mem_cache', 'container-memory-cache'), ('rx_bytes', 'container-network-receive-bytes-total'),
    ('tx_bytes', 'container-network-transmit-bytes-total'), ('rx_packets', 'container-network-receive-packets-total'),
    ('tx_packets', 'container-network-transmit-packets-total'), ('rx_errors', 'container-network-receive-errors-total'),
    ('tx_errors', 'container-network-transmit-errors-total'), ('fs_read', 'container-fs-reads-bytes-total'),
    ('fs_write', 'container-fs-writes-bytes-total'), ('sockets', 'container-sockets'),
    ('quota', 'container-spec-cpu-quota'), ('mem_limit', 'container-spec-memory-limit-bytes'),
    ('thr_periods', 'container-cpu-cfs-throttled-periods-total'), ('cfs_periods', 'container-cpu-cfs-periods-total'),
    ('thr_seconds', 'container-cpu-cfs-throttled-seconds-total')]     # cAdvisor: thoi gian bi bop CPU (CFS throttling)
RAW_APP = ['app-request-total', 'app-error-total', 'app-latency-50', 'app-latency-90', 'app-latency-95', 'app-latency-99']


# ------------------------------------------------------------------ docker api
class _UnixConn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost')
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect(self._path)


def docker_raw(path):
    c = _UnixConn('/var/run/docker.sock')
    try:
        c.request('GET', path)
        return c.getresponse().read()
    finally:
        c.close()


def docker_get(path):
    return json.loads(docker_raw(path))


def discover(project):
    flt = json.dumps({'label': [f'com.docker.compose.project={project}'], 'status': ['running']})
    out = {}
    for c in docker_get('/containers/json?filters=' + urllib.request.quote(flt)):
        svc = c['Labels'].get('com.docker.compose.service')
        if svc:
            out[svc] = c['Id']
    return out


# ------------------------------------------------------------------ cac ham thuan (kiem thu duoc khong can docker)
def parse_stats(s):
    """/containers/{id}/stats -> dict bo dem tich luy. Chiu ca cgroup v1 va v2."""
    cpu = s['cpu_stats']['cpu_usage']
    ms = s.get('memory_stats', {})
    st = ms.get('stats') or {}
    usage = ms.get('usage', 0)
    o = {'cpu_total': cpu['total_usage'] / 1e9, 'cpu_user': cpu.get('usage_in_usermode', 0) / 1e9,
         'cpu_system': cpu.get('usage_in_kernelmode', 0) / 1e9, 'mem_usage': usage,
         'mem_ws': usage - st.get('inactive_file', st.get('total_inactive_file', 0)),
         'mem_rss': st.get('anon', st.get('rss', st.get('total_rss', 0))),
         'mem_cache': st.get('file', st.get('cache', st.get('total_cache', 0)))}
    for k in ('rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets', 'rx_errors', 'tx_errors'):
        o[k] = sum((n or {}).get(k, 0) for n in (s.get('networks') or {}).values())
    r = w = 0
    for e in ((s.get('blkio_stats') or {}).get('io_service_bytes_recursive') or []):
        op = str(e.get('op', '')).lower()
        r += e.get('value', 0) if op == 'read' else 0
        w += e.get('value', 0) if op == 'write' else 0
    o['fs_read'], o['fs_write'] = r, w
    th = (s.get('cpu_stats') or {}).get('throttling_data') or {}
    o['thr_periods'], o['cfs_periods'] = th.get('throttled_periods', 0), th.get('periods', 0)
    o['thr_seconds'] = th.get('throttled_time', 0) / 1e9
    return o


def parse_sockstat(text):
    """'sockets: used N' -> N (so socket dang mo, nhu cAdvisor container_sockets)."""
    m = re.search(r'sockets:\s+used\s+(\d+)', text)
    return int(m.group(1)) if m else NAN


def limits_from_inspect(j, ncpu=0):
    """Gioi han CPU theo don vi K8s/cAdvisor (quota/100000 = core). NanoCpus >= NCPU cua VM = khong tran
    (docker khong go duoc NanoCpus bang `update`, nen 'khong tran' duoc bieu dien bang --cpus=NCPU)."""
    hc = j.get('HostConfig', {})
    nano = hc.get('NanoCpus') or 0
    if ncpu and nano >= ncpu * 1e9 * 0.999:
        nano = 0
    if nano:
        quota = nano / 1e9 * 100000
    else:
        quota = hc.get('CpuQuota') or 0
    return max(float(quota), 0.0), float(hc.get('Memory') or 0), j.get('State', {}).get('Pid')   # -1 (khong tran) -> 0 nhu cAdvisor


def demux_logs(buf):
    """Luong log docker (non-TTY): khung 8 byte [type,0,0,0,size BE] + payload -> danh sach dong."""
    lines, i = [], 0
    while i + 8 <= len(buf):
        size = struct.unpack('>I', buf[i + 4:i + 8])[0]
        payload = buf[i + 8:i + 8 + size]
        i += 8 + size
        lines.extend(payload.decode('utf-8', 'replace').splitlines())
    return lines


def parse_log_line(line):
    """'2024-01-18T16:30:59.012034902Z msg' -> (ns_epoch, msg) hoac None."""
    m = re.match(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z\s?(.*)$', line)
    if not m:
        return None
    ns = calendar.timegm(time.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S')) * 10 ** 9 + int((m.group(2) or '0').ljust(9, '0')[:9])
    return ns, m.group(3)


def keep_log(svc, msg):
    """Giu MOI dong log (RE2-SS giu ca dong khong phai access-log cua front-end); chi bo dong rong."""
    return bool(msg.strip())


def scrape_body(body):
    """Van ban Prometheus -> {'count','sum','err','buckets','routes'} (bo route metrics)."""
    agg = {'count': 0.0, 'sum': 0.0, 'err': 0.0, 'buckets': {}, 'routes': {}}
    for line in body.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        kind, lb, val = m.group(1), dict(_LABEL.findall(m.group(2))), float(m.group(3))
        if 'metrics' in lb.get('route', ''):
            continue
        sc = lb.get('status_code', '0')
        if kind.endswith('_count'):
            agg['count'] += val
            if sc[:1] == '5':
                agg['err'] += val
            key = (lb.get('method', '').upper(), lb.get('route', ''), sc[:1] + 'xx')
            agg['routes'].setdefault(key, [0.0, 0.0])[0] += val
        elif kind.endswith('_sum'):
            agg['sum'] += val
            key = (lb.get('method', '').upper(), lb.get('route', ''), sc[:1] + 'xx')
            agg['routes'].setdefault(key, [0.0, 0.0])[1] += val
        else:
            le = float('inf') if lb['le'] == '+Inf' else float(lb['le'])
            agg['buckets'][le] = agg['buckets'].get(le, 0.0) + val
    return agg


def quantile(q, d_buckets):
    if not d_buckets:
        return NAN
    les = sorted(d_buckets)
    total = d_buckets[les[-1]]
    if total <= 0:
        return NAN
    rank, lo_le, lo_n = q * total, 0.0, 0.0
    for le in les:
        n = d_buckets[le]
        if n >= rank:
            if math.isinf(le):
                return lo_le
            return lo_le + (le - lo_le) * ((rank - lo_n) / (n - lo_n) if n > lo_n else 0.0)
        lo_le, lo_n = le, n
    return lo_le


def vm_sample():
    with open('/proc/stat') as f:
        p = list(map(int, f.readline().split()[1:]))
    avail = 0
    with open('/proc/meminfo') as f:
        for l in f:
            if l.startswith('MemAvailable'):
                avail = int(l.split()[1]) / 1024.0
                break
    return p[3] + p[4], sum(p), avail


# ------------------------------------------------------------------ doc du lieu song
def read_container(cid, inspect_cache, now):
    try:
        s = parse_stats(docker_get(f'/containers/{cid}/stats?stream=false&one-shot=true'))
        ic = inspect_cache.get(cid)
        if ic is None or now - ic[0] > 10:
            ic = (now, limits_from_inspect(docker_get(f'/containers/{cid}/json'), NCPU[0]))
            inspect_cache[cid] = ic
        quota, mem_limit, pid = ic[1]
        s['quota'], s['mem_limit'] = quota, mem_limit
        s['sockets'] = NAN
        if pid:
            try:
                with open(f'/proc/{pid}/net/sockstat') as f:
                    s['sockets'] = parse_sockstat(f.read())
            except Exception:
                pass
        return s
    except Exception:
        return None


def scrape(svc):
    try:
        body = urllib.request.urlopen(f'http://{svc}:{METRICS_PORT[svc]}/metrics', timeout=4).read().decode('utf-8', 'replace')
    except Exception:
        return None
    return scrape_body(body)


def selftest():
    v2 = {'cpu_stats': {'cpu_usage': {'total_usage': 3e9, 'usage_in_usermode': 2e9, 'usage_in_kernelmode': 1e9}},
          'memory_stats': {'usage': 1000, 'stats': {'inactive_file': 100, 'anon': 500, 'file': 300}},
          'networks': {'eth0': {'rx_bytes': 10, 'tx_bytes': 20, 'rx_packets': 1, 'tx_packets': 2, 'rx_errors': 0, 'tx_errors': 0}},
          'blkio_stats': {'io_service_bytes_recursive': [{'op': 'read', 'value': 5}, {'op': 'Write', 'value': 7}]}}
    p = parse_stats(v2)
    assert p['cpu_total'] == 3 and p['mem_ws'] == 900 and p['mem_rss'] == 500 and p['mem_cache'] == 300
    assert p['rx_bytes'] == 10 and p['fs_read'] == 5 and p['fs_write'] == 7, p
    assert p['thr_periods'] == 0 and p['thr_seconds'] == 0
    v2['cpu_stats']['throttling_data'] = {'periods': 100, 'throttled_periods': 40, 'throttled_time': 2500000000}
    t = parse_stats(v2)
    assert t['thr_periods'] == 40 and t['cfs_periods'] == 100 and t['thr_seconds'] == 2.5, t
    v1 = {'cpu_stats': {'cpu_usage': {'total_usage': 1e9}}, 'memory_stats': {'usage': 50, 'stats': {'total_inactive_file': 10, 'rss': 20, 'cache': 5}}}
    q = parse_stats(v1)
    assert q['mem_ws'] == 40 and q['mem_rss'] == 20 and q['mem_cache'] == 5 and q['rx_bytes'] == 0
    assert parse_sockstat('sockets: used 123\nTCP: inuse 4 orphan 0') == 123 and math.isnan(parse_sockstat('x'))
    assert limits_from_inspect({'HostConfig': {'NanoCpus': 500000000, 'Memory': 0}, 'State': {'Pid': 7}}) == (50000.0, 0.0, 7)
    assert limits_from_inspect({'HostConfig': {}, 'State': {}}) == (0.0, 0.0, None)
    assert limits_from_inspect({'HostConfig': {'NanoCpus': 12000000000}, 'State': {'Pid': 3}}, 12) == (0.0, 0.0, 3)   # =NCPU -> khong tran
    assert limits_from_inspect({'HostConfig': {'NanoCpus': 500000000}, 'State': {}}, 12)[0] == 50000.0
    buf = b''.join(struct.pack('>BxxxI', 1, len(x)) + x for x in (b'2024-01-18T16:30:59.012034902Z GET /catalogue 200 6.1 ms - -\n',
                                                               b'2024-01-18T16:30:59Z caller=x took=1ms\n'))
    ln = demux_logs(buf)
    a, b = parse_log_line(ln[0]), parse_log_line(ln[1])
    assert a == (1705595459012034902, 'GET /catalogue 200 6.1 ms - -'), a
    assert b == (1705595459000000000, 'caller=x took=1ms'), b
    assert keep_log('front-end', a[1]) and keep_log('front-end', 'Posting Address: {}') and not keep_log('user', '  ')
    assert _ACCESS.match(a[1])
    assert _ANSI.sub('', '\x1b[0mPOST /register \x1b[32m200 \x1b[0m240.688 ms - 33\x1b[0m') == 'POST /register 200 240.688 ms - 33'
    body = ('request_duration_seconds_count{method="GET",route="/catalogue",status_code="200"} 10\n'
            'request_duration_seconds_sum{method="GET",route="/catalogue",status_code="200"} 0.5\n'
            'request_duration_seconds_count{method="POST",route="/{a}/{b}",status_code="500"} 2\n'
            'request_duration_seconds_count{method="GET",route="metrics",status_code="200"} 99\n'
            'request_duration_seconds_bucket{method="GET",route="/catalogue",status_code="200",le="0.005"} 4\n'
            'request_duration_seconds_bucket{method="GET",route="/catalogue",status_code="200",le="+Inf"} 10\n')
    g = scrape_body(body)
    assert g['count'] == 12 and g['err'] == 2 and g['routes'][('GET', '/catalogue', '2xx')] == [10.0, 0.5], g
    assert g['buckets'][0.005] == 4 and ('GET', 'metrics', '2xx') not in g['routes']
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='sockshop')
    ap.add_argument('--interval', type=float, default=5.0)
    ap.add_argument('--out', help='duong dan collector_<tag>.csv; raw_/logs_/routes_ dung chung tien to')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out:
        sys.exit('--out la bat buoc')

    NCPU[0] = int(docker_get('/info').get('NCPU', 0))
    ctrs = discover(a.project)
    svcs = sorted(ctrs)
    if not svcs:
        sys.exit('khong thay container nao cua project ' + a.project)
    web = [s for s in svcs if s in METRICS_PORT]
    print(f'[collector] {len(svcs)} container, {len(web)} co /metrics, interval={a.interval}s', flush=True)
    d_out = a.out
    r_out, l_out, u_out = (d_out.replace('collector_', p + '_', 1) for p in ('raw', 'logs', 'routes'))

    cols = ['time', 'vm_cpu_util', 'vm_mem_avail_mb']
    rcols = ['time']
    for s in svcs:
        cols += [f'{s}_cpu', f'{s}_mem', f'{s}_socket', f'{s}_diskio']
        rcols += [f'{s}_{n}' for _, n in RAW_CONT]
        if s in METRICS_PORT:
            cols += [f'{s}_workload', f'{s}_latency-50', f'{s}_latency-90', f'{s}_latency-95',
                     f'{s}_latency-99', f'{s}_latency-mean', f'{s}_error']
            rcols += [f'{s}_{n}' for n in RAW_APP]
    pool = ThreadPoolExecutor(max_workers=len(svcs) * 2 + len(web))
    inspect_cache, last_log_ns = {}, {}
    warned_sock = [False]

    def snapshot():
        t = time.time()
        cf = {s: pool.submit(read_container, ctrs[s], inspect_cache, t) for s in svcs}
        mf = {s: pool.submit(scrape, s) for s in web}
        return (t, {s: f.result() for s, f in cf.items()}, {s: f.result() for s, f in mf.items()}, vm_sample())

    def fetch_logs(s, since_s):
        try:
            buf = docker_raw(f'/containers/{ctrs[s]}/logs?stdout=1&stderr=1&timestamps=1&since={int(since_s)}')
        except Exception:
            return s, []
        out = []
        for ln in demux_logs(buf):
            pl = parse_log_line(ln)
            if not pl or pl[0] <= last_log_ns.get(s, 0):
                continue
            msg = _ANSI.sub('', pl[1])
            if keep_log(s, msg):
                out.append((pl[0], msg[:LOG_MSG_MAX]))
        return s, out[-LOG_LINES_PER_TICK:]

    prev = snapshot()
    with open(d_out, 'w', newline='') as fd, open(r_out, 'w', newline='') as fr, \
            open(l_out, 'w', newline='', encoding='utf-8') as fl, open(u_out, 'w', newline='') as fu:
        wd, wr, wl, wu = csv.writer(fd), csv.writer(fr), csv.writer(fl), csv.writer(fu)
        wd.writerow(cols)
        wr.writerow(rcols)
        wl.writerow(['time', 'timestamp', 'container_name', 'message', 'level', 'req_path', 'error'])
        wu.writerow(['time', 'service', 'method', 'route', 'status_class', 'rps', 'mean_ms'])
        nxt = time.time() + a.interval
        t_prev_logs = time.time()
        while True:
            time.sleep(max(0.0, nxt - time.time()))
            nxt += a.interval
            cur = snapshot()
            dt = cur[0] - prev[0]
            row, raw = {'time': round(cur[0], 3)}, {'time': round(cur[0], 3)}
            di, dtot = cur[3][0] - prev[3][0], cur[3][1] - prev[3][1]
            row['vm_cpu_util'] = 1.0 - di / dtot if dtot > 0 else NAN
            row['vm_mem_avail_mb'] = cur[3][2]
            for s in svcs:
                c0, c1 = prev[1].get(s), cur[1].get(s)
                if c1:
                    for key, name in RAW_CONT:
                        raw[f'{s}_{name}'] = c1.get(key, '')
                if c0 and c1:
                    row[f'{s}_cpu'] = (c1['cpu_total'] - c0['cpu_total']) / dt * 100.0
                    row[f'{s}_mem'] = c1['mem_ws']
                    row[f'{s}_socket'] = c1['sockets']
                    row[f'{s}_diskio'] = ((c1['fs_read'] + c1['fs_write']) - (c0['fs_read'] + c0['fs_write'])) / dt
                    if math.isnan(c1['sockets']) and not warned_sock[0]:
                        warned_sock[0] = True
                        print('[collector] CANH BAO: khong doc duoc /proc/<pid>/net/sockstat (can --pid=host --cap-add SYS_PTRACE); _socket=NaN', flush=True)
                if s not in METRICS_PORT:
                    continue
                m0, m1 = prev[2].get(s), cur[2].get(s)
                if not (m0 and m1):
                    continue
                dc = m1['count'] - m0['count']
                if dc < 0:      # service restart -> bo cua so nay
                    continue
                row[f'{s}_workload'] = dc / dt
                row[f'{s}_error'] = (m1['err'] - m0['err']) / dt
                db = {le: m1['buckets'].get(le, 0.0) - m0['buckets'].get(le, 0.0) for le in m1['buckets']}
                lat = {q: quantile(q / 100.0, db) for q in (50, 90, 95, 99)}
                for q, v in lat.items():
                    row[f'{s}_latency-{q}'] = v
                row[f'{s}_latency-mean'] = (m1['sum'] - m0['sum']) / dc if dc > 0 else NAN
                raw[f'{s}_app-request-total'] = m1['count']
                raw[f'{s}_app-error-total'] = m1['err']
                for q, v in lat.items():
                    raw[f'{s}_app-latency-{q}'] = v
                for key, (cnt, sm) in m1['routes'].items():
                    p_cnt, p_sm = m0['routes'].get(key, (0.0, 0.0))
                    dr = cnt - p_cnt
                    if dr > 0:
                        wu.writerow([round(cur[0], 3), s, key[0], key[1], key[2], round(dr / dt, 4),
                                     round((sm - p_sm) / dr * 1000.0, 3)])
            wd.writerow([row.get(c, '') for c in cols])
            wr.writerow([raw.get(c, '') for c in rcols])
            # log: truy van tung container (song song), ghi theo thoi gian tang dan
            res = list(pool.map(lambda s: fetch_logs(s, t_prev_logs - 1), svcs))
            t_prev_logs = cur[0]
            merged = sorted((ns, s, msg) for s, lines in res for ns, msg in lines)
            for ns, s, msg in merged:
                last_log_ns[s] = max(last_log_ns.get(s, 0), ns)
                wl.writerow([time.strftime('%H:%M', time.gmtime(ns / 1e9)), ns, s, msg, '', '', ''])
            for f in (fd, fr, fl, fu):
                f.flush()
            prev = cur


if __name__ == '__main__':
    main()
