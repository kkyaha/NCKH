# -*- coding: utf-8 -*-
"""DOI CHIEU HE A/HE B theo DUNG CHAIN cua tung request -- lop ket hop hai he
ma KHONG gop co che du doan cua chung.

VI SAO FILE NAY TON TAI
========================
`papers/p1_du_phong/experiments/feasibility/doi_chieu_hai_he.py` tinh MOT bang
tinh (lech_u_pct) toan cuc cho 7 service, chay tay, khong gan vao pipeline.
Ban than con so toan cuc (max qua 5 node SCORED = 0,301%, tu node `user`) la
mot can tren THUA cho moi request KHONG cham node `user` -- vi du GET_CATALOGUE
chi cham front-end/catalogue, ma van bi gan nhan "co the lech toi 0,301%".

Ham `cross_check_for_chain()` o day doc lai dung bang da tinh (CSV, khong tinh
lai mo hinh o request-time -- re, khong doi hanh vi cua He A/He B), roi LOC
theo dung service co trong chain cua request + nam trong SCORED cua He B. Hai
service KHONG nam trong SCORED (`payment`, `shipping`) bi loai co chu dich --
xem papers/p1_du_phong/docs/DATA_FRAMEWORK.md muc 3: CPU qua nho (payment 0,1
core) lam sai so tuong doi vo nghia, va shipping xu ly bat dong bo qua hang doi
-- KHONG phai lo hong do phu can He A "lap vao". Thu lay He A lap hai node nay
da bi loai o nhung lan thao luan truoc (xem HE_THONG.md/paper_draft.tex): tin
hieu CPU cua He A tren hai node do nhiem DUNG cai nhieu da khien chung bi loai
khoi SCORED, nen "lap" se dua nhiem do ngay lai vao phan quyet.

Khong doi `verdict` cua He B -- day CHI la mot nhan do tin kem theo, loc dung
pham vi co co so (SCORED), khong mo rong pham vi dung duoc cua mot tin hieu da
biet la nhieu.
"""
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
while BASE_DIR != os.path.dirname(BASE_DIR) and not os.path.isdir(os.path.join(BASE_DIR, 'src')):
    BASE_DIR = os.path.dirname(BASE_DIR)
assert os.path.isdir(os.path.join(BASE_DIR, 'src')), 'khong tim thay goc repo (thu muc chua src/)'

DOI_CHIEU_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'scm_results', 'doi_chieu_hai_he.csv')
SCORED = {'front-end', 'catalogue', 'user', 'carts', 'orders'}


def load_lech_u_table(path: str = DOI_CHIEU_CSV) -> dict:
    """{service: lech_u_pct} tu CSV da tinh boi doi_chieu_hai_he.py.

    Khong tu tinh lai mo hinh -- neu CSV chua co, bao loi ro rang thay vi
    doan mot gia tri, vi day la dau vao cho mot nhan do TIN, sai nhan do tin
    con te hon khong co nhan do tin.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"khong thay {path} -- chay "
            "papers/p1_du_phong/experiments/feasibility/doi_chieu_hai_he.py truoc")
    table = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            table[row['service']] = float(row['lech_u_pct'])
    return table


def cross_check_for_chain(services, table: dict = None, scored=SCORED) -> dict:
    """Nhan do tin cho DUNG chain cua mot request.

    services: danh sach service trong chain cua request nay (tu ParserAgent/
        SOCKSHOP_CALL_CHAINS), KHONG phai toan bo 7 service.
    Tra ve {'max_lech_u_pct': float|None, 'per_service': {...}, 'excluded': [...]}
    -- 'excluded' la cac service trong chain nhung KHONG nam trong `scored`
    (payment, shipping): van duoc He A/He B lan truyen tai, nhung khong duoc
    tinh vao nhan do tin hay vao phan quyet, dung ly do da neu o SCORED.
    """
    if table is None:
        table = load_lech_u_table()
    in_scope = [s for s in services if s in scored]
    excluded = [s for s in services if s not in scored]
    per_service = {s: table[s] for s in in_scope if s in table}
    max_lech = max(per_service.values()) if per_service else None
    return {'max_lech_u_pct': max_lech, 'per_service': per_service, 'excluded': excluded}
