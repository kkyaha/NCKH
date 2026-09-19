# -*- coding: utf-8 -*-
"""
CALL CHAIN MINING TU LOG THO (system-agnostic)
================================================
Tong quat hoa buoc "services" cua mot archetype (request type) tu LOG THO
(khong phai tu span/trace) -- bo sung cho hai module tong quat da co:

  - src/graph/extract_graph_generic.py : do thi phu thuoc TOAN CUC tu
    span-level tracing (spanID/parentSpanID/serviceName).
  - src/scm/taxonomy_builder.py        : suy `services` cua TUNG archetype
    bang bounded BFS tu mot seed service viet tay tren do thi tren -- xep
    "semi-automatic" trong bang "General Specification" cua paper vi seed
    van phai chon tay.

Module nay di THANG tu log dong (khong can do thi phu thuoc lam trung
gian): voi he thong co gateway ghi log kieu "access log" chuan (Express/
Morgan-style: "<METHOD> <path> <status> <duration> ms ..."), ta phat hien
tung request THAT qua dong log gateway, roi nhin cac dong log CUA DICH VU
KHAC xay ra trong khoang [T - duration, T] ngay truoc no -- do la nhung
dich vu THAT SU tham gia phuc vu request do. Gop nhieu lan quan sat cua
cung mot (method, path) cho ra tan suat xuat hien cua tung service, tu do
chon ra services CUA ARCHETYPE do MA KHONG CAN SEED VIET TAY -- nang cap
"semi-automatic" (BFS tu seed) len gan "fully automatic" (quan sat that)
khi he thong co du lieu log dang nay.

Gia dinh DUY NHAT (khong gia dinh ten dich vu/route cu the nao):
  1. logs.csv (hoac tuong duong) co it nhat 3 cot: mot cot thoi gian co
     the sap xep (vd `timestamp`, don vi bat ky mien don dieu tang), mot
     cot ten container/service phat ra dong log (`container_name`), va
     mot cot noi dung (`message`).
  2. It nhat MOT service (thuong la gateway/front-end) ghi access-log
     dang "<METHOD> <path> <status> <duration>ms" cho moi request no xu
     ly xong -- day la dinh dang pho bien (Express morgan, nginx, Spring
     access log deu bien the gan giong) nen KHONG rieng cho SockShop.

He thong KHONG co dang log nay (vd chi co structured trace/span, khong co
access-log dong gateway) thi dung extract_graph_generic.py + taxonomy_
builder.py thay the -- hai huong bo sung nhau, khong thay the nhau.

Da kiem chung thuc nghiem tren toan bo 90 (scenario x run) cua RE2-SS
(SockShop that): xem experiments/callchain_from_logs_validation.py -- so
sanh services mine duoc tu log voi SOCKSHOP_CALL_CHAINS (bang tay hien co
trong request_router.py) lam bang chung do chinh xac, khong chi chay thu.
"""

import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from scm_edge_selector import _knee_point_index  # noqa: E402


# Regex tong quat cho dong access-log dang "METHOD /path STATUS DURATION ms":
# khong neo ten dich vu/route nao -- chi doi hoi 4 nhom (method, path, status,
# duration) theo dung thu tu pho bien cua Express morgan / nginx / tuong tu.
_ACCESS_LOG_RE = re.compile(
    r'^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(\S+)\s+(\d{3})\s+([\d.]+)\s*ms',
    re.IGNORECASE,
)


def load_logs(logs_csv_path: str, time_col: str = 'timestamp',
               container_col: str = 'container_name',
               message_col: str = 'message') -> pd.DataFrame:
    """Doc logs.csv, chi giu 3 cot can thiet, sap xep theo thoi gian.

    Chi doi hoi 3 cot ton tai (ten cot truyen duoc qua tham so) -- khong
    doc bat ky cot dac thu he thong nao khac (vd `req_path` cua RE2-SS bi
    bo qua vi thuong rong, xem docstring module).
    """
    df = pd.read_csv(logs_csv_path, usecols=[time_col, container_col, message_col])
    df = df.rename(columns={time_col: 'timestamp', container_col: 'container_name',
                             message_col: 'message'})
    df = df.dropna(subset=['timestamp', 'container_name', 'message'])
    return df.sort_values('timestamp').reset_index(drop=True)


def detect_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Tim moi dong log khop dang access-log "METHOD /path STATUS DURATION ms".

    Khong gia dinh container nao la "gateway" -- BAT KY container nao phat
    ra dong dang nay duoc coi la mot entry point (ho tro ca he nhieu
    gateway nhu Train Ticket, xem taxonomies/trainticket.py docstring ve
    14 node type=='gateway').

    Tra ve DataFrame [row_idx, timestamp, gateway, method, path, duration_ms].
    """
    rows = []
    for i, msg in enumerate(df['message'].astype(str)):
        m = _ACCESS_LOG_RE.match(msg)
        if not m:
            continue
        rows.append({
            'row_idx':     i,
            'timestamp':   df.at[i, 'timestamp'],
            'gateway':     df.at[i, 'container_name'],
            'method':      m.group(1).upper(),
            'path':        m.group(2),
            'duration_ms': float(m.group(4)),
        })
    return pd.DataFrame(rows)


def mine_service_cooccurrence(df: pd.DataFrame, endpoints_df: pd.DataFrame,
                               duration_multiplier: float = 1.5,
                               min_window_ns: int = 2_000_000,
                               label_fn: Optional[Callable[[str, str], str]] = None
                               ) -> pd.DataFrame:
    """Voi moi lan quan sat mot endpoint (dong access-log), lay cua so THOI
    GIAN [T - duration*multiplier - buffer, T] va gom tap container co hoat
    dong log trong cua so do -- day la tap dich vu THAT SU tham gia phuc vu
    request do (khong phai suy dien qua do thi).

    Dung chinh `duration_ms` cua tung request lam do rong cua so (thay vi
    mot hang so co dinh) LA DIEM MAU CHOT: he thong nhieu request/giay se
    co request khac chen vao giua neu dung cua so co dinh qua rong, lam
    "ro" ket qua bang cac dich vu khong lien quan (da kiem chung truc tiep
    tren RE2-SS: cua so co dinh 300ms cho non-tuong quan ~30% cho nhung
    service khong lien quan, cua so theo duration rieng cua tung request
    dua no ve <10% nhieu -- xem experiments/callchain_from_logs_validation.py).

    duration_multiplier: he so nhan them vao duration bao cao (vi bao thoi
    gian tra ve front-end co the ngan hon tong thoi gian goi noi bo do
    logging/flush delay) -- 1.5 la gia tri kiem chung thuc nghiem tot,
    KHONG phai hang so ly thuyet.
    min_window_ns: san toi thieu cua cua so (bu clock skew giua container
    khi duration bao cao rat nho, vd <1ms).

    Tra ve DataFrame [endpoint, service, n_cooccur, n_total, frequency].
    """
    if endpoints_df.empty:
        return pd.DataFrame(columns=['endpoint', 'service', 'n_cooccur', 'n_total', 'frequency'])

    ts = df['timestamp'].values
    containers = df['container_name'].values
    label_fn = label_fn or (lambda method, path: f'{method} {path}')

    agg: Dict[str, Counter] = defaultdict(Counter)
    counts: Counter = Counter()

    for row in endpoints_df.itertuples():
        endpoint = label_fn(row.method, row.path)
        window_ns = int(row.duration_ms * 1e6 * duration_multiplier) + min_window_ns
        lo = row.timestamp - window_ns
        mask = (ts >= lo) & (ts <= row.timestamp)
        touched = set(containers[mask])
        agg[endpoint].update(touched)
        counts[endpoint] += 1

    out_rows = []
    for endpoint, ctr in agg.items():
        n_total = counts[endpoint]
        for service, n_cooccur in ctr.items():
            out_rows.append({
                'endpoint':   endpoint,
                'service':    service,
                'n_cooccur':  n_cooccur,
                'n_total':    n_total,
                'frequency':  round(n_cooccur / n_total, 4),
            })
    return pd.DataFrame(out_rows)


def select_call_chain_services(freq_df: pd.DataFrame, endpoint: str,
                                min_frequency_floor: float = 0.05) -> List[str]:
    """Chon services cua MOT endpoint bang diem khuyu tay tren tan suat da
    sap xep giam dan -- dung LAI _knee_point_index() cua scm_edge_selector.py
    thay vi viet mot ham gan giong het, giu MOT quy tac khuyu tay duy nhat
    cho toan bo repo (canh SCM lan services archetype).

    min_frequency_floor: san chan duoi tuyet doi (mac dinh 5%) de loai
    service bi "ro" vao cua so do request lien ke/trung lap gay ra, ngay ca
    khi no nam truoc diem khuyu tay -- cung tinh than voi `min_gain` trong
    select_edges_by_knee_point().

    Tra ve danh sach service, sap xep giam dan theo tan suat (gateway/entry
    thuong la 1.00, dung dau danh sach mot cach tu nhien).
    """
    sub = freq_df[freq_df['endpoint'] == endpoint].sort_values('frequency', ascending=False)
    sub = sub[sub['frequency'] > min_frequency_floor]
    if sub.empty:
        return []
    knee_idx = _knee_point_index(sub['frequency'].values)
    return sub['service'].iloc[:knee_idx + 1].tolist()


def mine_call_chains_from_logs(logs_csv_path: str, duration_multiplier: float = 1.5,
                                min_frequency_floor: float = 0.05,
                                label_fn: Optional[Callable[[str, str], str]] = None
                                ) -> Dict[str, Dict]:
    """Chay ca 3 buoc (load -> detect endpoint -> co-occurrence -> chon
    services) thanh MOT loi goi, cho MOT file logs.csv. Dung
    mine_call_chains_from_multiple() de gop nhieu run/scenario (khuyen
    nghi -- mot run don co the co qua it lan lap lai mot endpoint hiem).

    Tra ve dict: {endpoint_label: {'services': [...], 'n_observations': int,
    'service_frequency': {service: frequency}}}.
    """
    df = load_logs(logs_csv_path)
    endpoints_df = detect_endpoints(df)
    return _finish_mining(df, endpoints_df, duration_multiplier,
                           min_frequency_floor, label_fn)


def mine_call_chains_from_multiple(logs_csv_paths: List[str],
                                    duration_multiplier: float = 1.5,
                                    min_frequency_floor: float = 0.05,
                                    label_fn: Optional[Callable[[str, str], str]] = None
                                    ) -> Dict[str, Dict]:
    """Nhu mine_call_chains_from_logs, nhung gop co-occurrence tren NHIEU
    file logs.csv (vd toan bo scenario/run cua benchmark) truoc khi chon
    services -- moi run xu ly DOC LAP (cua so thoi gian khong bao gio bat
    qua ranh gioi 2 file khac nhau), chi cong don Counter sau cung. Cho ket
    qua on dinh hon 1 run don le, dac biet voi endpoint hiem.
    """
    all_freq = []
    for path in logs_csv_paths:
        df = load_logs(path)
        endpoints_df = detect_endpoints(df)
        f = mine_service_cooccurrence(df, endpoints_df, duration_multiplier,
                                       label_fn=label_fn)
        all_freq.append(f)

    if not all_freq or all(f.empty for f in all_freq):
        return {}

    combined = pd.concat(all_freq, ignore_index=True)
    merged = combined.groupby(['endpoint', 'service'], as_index=False).agg(
        n_cooccur=('n_cooccur', 'sum'), n_total=('n_total', 'sum'))
    merged['frequency'] = (merged['n_cooccur'] / merged['n_total']).round(4)

    return _chains_from_freq_df(merged, min_frequency_floor)


def _finish_mining(df, endpoints_df, duration_multiplier, min_frequency_floor, label_fn):
    freq_df = mine_service_cooccurrence(df, endpoints_df, duration_multiplier, label_fn=label_fn)
    return _chains_from_freq_df(freq_df, min_frequency_floor)


def _chains_from_freq_df(freq_df: pd.DataFrame, min_frequency_floor: float) -> Dict[str, Dict]:
    result = {}
    for endpoint in freq_df['endpoint'].unique():
        services = select_call_chain_services(freq_df, endpoint, min_frequency_floor)
        if not services:
            continue
        sub = freq_df[freq_df['endpoint'] == endpoint]
        n_obs = int(sub['n_total'].iloc[0])
        freqs = dict(zip(sub['service'], sub['frequency']))
        result[endpoint] = {
            'services':          services,
            'n_observations':    n_obs,
            'service_frequency': {s: freqs[s] for s in services},
        }
    return result


def merge_with_archetype_meta(mined: Dict[str, Dict], archetype_meta: Dict[str, Dict]) -> Dict[str, Dict]:
    """Ghep ket qua mine_call_chains_from_* (services tu dong) voi phan
    VIET TAY con lai cua mot archetype (keywords/description/resource_
    profile/expected_delta_pct -- "assisted"/"not automatable" theo bang
    General Specification, khong the suy tu log).

    archetype_meta: {endpoint_label: {'keywords','description',
    'resource_profile','expected_delta_pct'}} -- CHI can khai bao cho
    nhung endpoint muon dua vao CALL_CHAINS cuoi cung (endpoint mine duoc
    nhung khong co trong archetype_meta se bi bo qua lang le, cho phep loc
    bot cac route noi bo/khong danh cho request khach hang).

    Tra ve dict CUNG SCHEMA voi SOCKSHOP_CALL_CHAINS / taxonomy_builder.
    build_call_chains() -- drop-in cho ParserAgent(call_chains=...).
    """
    out = {}
    for label, meta in archetype_meta.items():
        if label not in mined:
            continue
        out[label] = {
            'services':           mined[label]['services'],
            'description':        meta['description'],
            'keywords':           meta['keywords'],
            'resource_profile':   meta['resource_profile'],
            'expected_delta_pct': meta['expected_delta_pct'],
        }
    return out


__all__ = [
    'load_logs', 'detect_endpoints', 'mine_service_cooccurrence',
    'select_call_chain_services', 'mine_call_chains_from_logs',
    'mine_call_chains_from_multiple', 'merge_with_archetype_meta',
]
