# -*- coding: utf-8 -*-
"""SINH MANIFEST DU LIEU -- ban do cua `data/raw/`.

Van de no giai: ten thu muc KHONG noi len vai tro. Co sau bien the `SS-LIMITS-*`,
mot `SS-PROSP2` chua chin thu muc con, va ba bo ten giong nhau nhung mot la train,
mot la test, mot chi la phep xac minh. Khong doc code thi khong phan biet duoc.

MOI VAI TRO DUOC SUY RA TU BANG CHUNG, khong gan tay:
  (1) co CLI trong ma nguon:  `--train-dir X` -> X la train;  `--ramp-dir X` -> X la test
  (2) o danh gia:             moi o trong p3_evaluation.csv / verdict_decisions.csv
                              duoc doi chieu voi thu muc chua `ramp_<o>` cua no
  (3) cau truc thu muc:       chi co `ramp_base`        -> phep do co so / xac minh
                              co `level_*`              -> hieu chuan u*
                              khong ramp, khong level   -> tham do tai
  (4) phia tham chieu:        chi duoc papers/p2 doc    -> benchmark ngoai cua bai 2

PHU THUOC NGUOC duoc lay bang cach doi chieu TUNG O trong moi file dong bang voi
thu muc chua ramp cua o do -- khong phai khop ten chuoi (ten bo du lieu khong he
xuat hien trong cac file dong bang).

CANH BAO DOI TEN -- da do, khong phai phong doan:
  `FP.sha256_files` bam `basename(dirname(dirname(p)))` + noi dung file, tuc TEN RAMP
  (`ramp_base`) nam trong hash, con ten thu muc bo du lieu thi KHONG. Hau qua:
    * doi ten `ramp_*`  -> `train_sha256`/`ramp_base_sha256` lech -> MAT kha nang xac minh
    * doi ten bo du lieu -> hash khong doi, nhung lam chet moi dong CLI da ghi trong
      docs/notebook va moi duong dan trong manifest nay
  Ca hai deu khong duoc lam. Ban do nay duoc THEM VAO, kho du lieu khong bi xep lai.

Chay:  python papers/p1_du_phong/experiments/collect/sinh_manifest_du_lieu.py
Ra:    data/manifest.json  (may doc)   +   data/DATA.md  (nguoi doc)
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *(['..'] * 4)))
from src.scm import feasibility_predictor as FP   # dung DUNG ham bam cua ban dang chay

_P = os.path.dirname(os.path.abspath(__file__))
while _P != os.path.dirname(_P) and not os.path.isdir(os.path.join(_P, 'src')):
    _P = os.path.dirname(_P)
RAW = os.path.join(_P, 'data', 'raw')
FROZEN = os.path.join(_P, 'data', 'processed', 'frozen')


def ramp_cua(d):
    """[(ten ramp, so run)] cua mot thu muc."""
    return [(os.path.basename(r), len([x for x in glob.glob(os.path.join(r, 'run*')) if os.path.isdir(x)]))
            for r in sorted(glob.glob(os.path.join(d, 'ramp_*')))]


def liet_ke_bo():
    """Moi bo du lieu, mo mot cap con khi thu muc cha chi la vo boc (nhu SS-PROSP2)."""
    res = {}
    for p in sorted(glob.glob(os.path.join(RAW, '*'))):
        if not os.path.isdir(p):
            continue
        ten, r = os.path.basename(p), ramp_cua(p)
        if r:
            res[ten] = r
            continue
        con = {f'{ten}/{os.path.basename(q)}': ramp_cua(q)
               for q in sorted(glob.glob(os.path.join(p, '*'))) if os.path.isdir(q) and ramp_cua(q)}
        res.update(con or {ten: []})
    return res


def co_cli():
    """(1) Co CLI: thu muc nao dung sau --train-dir / --ramp-dir."""
    vt = defaultdict(set)
    goc = {'--train-dir': 'train', '--ramp-dir': 'test'}
    for f in glob.glob(os.path.join(_P, 'papers', '**', '*'), recursive=True):
        if os.path.splitext(f)[1] not in ('.py', '.md', '.sh', '.ipynb'):
            continue
        try:
            t = open(f, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        for flag, d in re.findall(r'(--train-dir|--ramp-dir)\s+(?:data/raw/)?([A-Za-z0-9_\-/]+)', t):
            vt[d.strip('/')].add(goc[flag])
    return vt


def hash_bo(ten):
    """(train_sha256, ramp_base_sha256) tinh lai tu dia, dung DUNG ham cua freeze_predictions:
      train  = <bo>/*/*/simple_metrics.csv
      base   = <bo>/ramp_base/run*/simple_metrics.csv
    Nhap ham tu src/scm de khong bao gio lech khoi ban dang dung that."""
    d = os.path.join(RAW, ten)
    tr = sorted(glob.glob(os.path.join(d, '*', '*', 'simple_metrics.csv')))
    ba = sorted(glob.glob(os.path.join(d, 'ramp_base', 'run*', 'simple_metrics.csv')))
    return (FP.sha256_files(tr) if tr else None, FP.sha256_files(ba) if ba else None)


def khop_hash(BO):
    """Doi chieu hash tren dia voi fingerprint cua tung file dong bang.

    Day la cach duy nhat gan dung: doi chieu theo TEN O that bai vi o `base` co trong
    MOI bo, va `cartsumx1` nam o hai bo (SS-LIMITS-INDEP va SS-LIMITS-CLEAN). Hash thi
    duy nhat. Doi chieu nay dong thoi la mot phep KIEM TOAN TOAN VEN: frozen nao khong
    khop bo nao tren dia la frozen khong con tai lap duoc.
    """
    H = {ten: hash_bo(ten) for ten in BO}
    tr_map = {h: t for t, (h, _) in H.items() if h}
    ba_map = {h: t for t, (_, h) in H.items() if h}
    vt, dep, mo_coi = defaultdict(set), defaultdict(set), []
    for f in sorted(glob.glob(os.path.join(FROZEN, 'predictions_frozen_*.json'))):
        try:
            fp = json.load(open(f, encoding='utf-8')).get('fingerprint') or {}
        except (OSError, ValueError):
            continue
        n, t, b = os.path.basename(f), tr_map.get(fp.get('train_sha256')), ba_map.get(fp.get('ramp_base_sha256'))
        if t:
            vt[t].add('train')
            dep[t].add(n)
        if b:
            vt[b].add('test')
            dep[b].add(n)
        if not t and not b:
            mo_coi.append(n)
    return vt, {k: sorted(v) for k, v in dep.items()}, mo_coi


def khop_co_che(BO):
    """Nhan dien + XAC MINH bo train bang cach KHOP LAI CO CHE, khong bang hash.

    Ly do phai lam the: `train_sha256` trong fingerprint KHONG tai lap duoc tu bat ky
    du lieu nao tren dia. Da loai sau gia thuyet (noi dung tep doi, tap tep doi, phien
    ban ham bam doi, dang duong dan, quy uoc ten thu muc muc tai, mtime) -- khong cai
    nao giai thich duoc. Nguyen nhan: CHUA XAC DINH.

    Nhung dieu quan trong hon hash van dung duoc: khop lai `fit_mechanism` tren bo
    nghi van va so voi `mechanism` da dong bang. Trung tung chu so (lam tron 5) tren
    ca 7 dich vu la bang chung manh hon hash -- hash chi chung minh byte giong nhau,
    con phep nay chung minh KET QUA tai lap duoc.

    Chi thu cac bo co layout train (`level_*`) de khong phai doc het 11 GB.
    """
    ung_vien = [t for t in BO if glob.glob(os.path.join(RAW, t, 'level_*'))]
    vt, dep = defaultdict(set), defaultdict(set)
    for f in sorted(glob.glob(os.path.join(FROZEN, 'predictions_frozen_*.json'))):
        try:
            j = json.load(open(f, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        mc, tm = j.get('mechanism'), (j.get('params') or {}).get('train_max_rps')
        if not mc or not tm:
            continue
        for t in ung_vien:
            try:
                d = FP.load_runs(os.path.join(RAW, t))
                m = FP.fit_mechanism(d[FP.level_of(d) <= tm])
            except Exception:
                continue
            if all(s in m and round(m[s]['alpha'], 5) == v['alpha']
                   and round(m[s]['beta'], 5) == v['beta'] for s, v in mc.items()):
                vt[t].add('train (xác minh bằng cơ chế)')
                dep[t].add(os.path.basename(f))
                break
    return vt, {k: sorted(v) for k, v in dep.items()}


def noi_chuoi_p3(DEP):
    """File P3 khong co fingerprint rieng -- no tro ve cha qua `base_frozen_file`.
    Noi chuoi de bo du lieu nhan duoc ca cac ket qua phu thuoc gian tiep."""
    cha = {}
    for f in sorted(glob.glob(os.path.join(FROZEN, 'predictions_frozen_*.json'))):
        try:
            j = json.load(open(f, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if j.get('base_frozen_file'):
            cha[os.path.basename(f)] = j['base_frozen_file']
    for bo, fs in DEP.items():
        them = {c for c, p in cha.items() if p in fs}
        DEP[bo] = sorted(set(fs) | them)
    return DEP


def o_cua_bo(ten):
    """Cac o danh gia ma bo nay chua ramp cho (nhieu bo co the chua cung mot o)."""
    return sorted({re.sub(r'_x(\d)$', r'x\1', os.path.basename(d)[len('ramp_'):])
                   for d in glob.glob(os.path.join(RAW, ten, 'ramp_*')) if os.path.isdir(d)} - {'base'})


def tap_danh_gia(BO):
    """Ten tap danh gia gan voi tung bo, qua o -- ghi ro khi mot o nam o nhieu bo."""
    p = os.path.join(FROZEN, 'p3_evaluation.csv')
    if not os.path.exists(p):
        return {}
    E = pd.read_csv(p).drop_duplicates('cell').set_index('cell')['set'].to_dict()
    return {ten: sorted({E[o] for o in o_cua_bo(ten) if o in E}) for ten in BO}


def lien_ket_qua_o(BO):
    """Noi bo TEST voi frozen qua O du doan. Bang chung YEU hon hash, nen duoc danh dau rieng.

    Phai co vi hash chi gan duoc bo train (khop co che) va SS-LIMITS (khop ramp_base).
    Cac bo test con lai -- CLEAN, INDEP, PROSP, PROSP2* -- co frozen voi `ramp_base_files: 0`
    hoac ramp_base khong khop, nen hash khong gan duoc chung. O thi gan duoc, keo theo
    nhap nhang khi mot o nam o nhieu bo; cho nen moi o nhap nhang deu duoc ghi ro.
    """
    o2bo = defaultdict(set)
    for ten in BO:
        for o in o_cua_bo(ten):
            o2bo[o].add(ten)
    lk, nhap_nhang = defaultdict(set), defaultdict(set)
    for f in sorted(glob.glob(os.path.join(FROZEN, 'predictions_frozen_*.json'))):
        try:
            j = json.load(open(f, encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for pr in (j.get('predictions') or []):
            o = f"{pr.get('feature')}x{int(pr.get('scale', 1))}"
            for ten in o2bo.get(o, ()):
                lk[ten].add(os.path.basename(f))
                if len(o2bo[o]) > 1:
                    nhap_nhang[ten].add(o)
    return ({k: sorted(v) for k, v in lk.items()},
            {k: sorted(v) for k, v in nhap_nhang.items()})


def ket_qua_bai_2(BO):
    """Noi bo du lieu cua BAI 2 voi cac CSV ket qua cua no.

    Can thiet vi cac file dong bang la cua BAI 1; khong co cai nao tro den RE2-SS,
    RE2-OB hay trainticket, nen neu chi xet frozen thi bon bo nay bi xep oan vao
    "khong ket qua nao phu thuoc" trong khi chung cho ra toan bo bai 2.

    Anh xa `he` -> thu muc KHONG gan tay: doc tu bang HE trong rq6_wrapper_rcaeval.py.
    """
    w = os.path.join(_P, 'papers', 'p2_khop_tang', 'experiments', 'rq6_wrapper_rcaeval.py')
    try:
        t = open(w, encoding='utf-8').read()
    except OSError:
        return {}
    anh_xa = {he: d for he, d in re.findall(
        r"'(\w+)':\s*dict\(graph=[^)]*?'data',\s*'raw',\s*'([A-Za-z0-9_\-]+)'", t)}
    if not anh_xa:
        return {}
    dem = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(os.path.join(_P, 'data', 'processed', 'scm_results', '*.csv'))):
        try:
            D = pd.read_csv(f, usecols=['he'])
        except (OSError, ValueError):
            continue
        for he, n in D.he.value_counts().items():
            if he in anh_xa:
                dem[anh_xa[he]][os.path.basename(f)] += int(n)
    # Mot so bo khong co cot `he` ma duoc dat ten thang vao ten tep ket qua
    # (vd `alibaba_*.csv`). Noi theo tien to ten -- la suy luan theo ten, yeu hon
    # cot `he`, nen chi dung khi cot `he` khong gan duoc bo do.
    chua_gan = [bo for bo in BO if not dem.get(bo)]      # chot truoc, khong xet lai trong vong lap
    for f in sorted(glob.glob(os.path.join(_P, 'data', 'processed', 'scm_results', '*.csv'))):
        b = os.path.basename(f)
        for bo in chua_gan:
            if b.lower().startswith(bo.lower() + '_'):
                try:
                    dem[bo][b] = sum(1 for _ in open(f, encoding='utf-8', errors='ignore')) - 1
                except OSError:
                    pass
    return {bo: dict(v) for bo, v in dem.items()}


def suy_vai_tro(ten, ramps, CLI, VT_HASH, VT_CC, TAP):
    """Hop bang chung, manh den yeu: hash noi dung > co CLI > o danh gia > cau truc."""
    goc = ten.split('/')[0]
    vt = set(VT_HASH.get(ten, set())) | set(VT_CC.get(ten, set()))     # (0) hash / khop lai co che
    vt |= CLI.get(ten, set()) | CLI.get(goc, set())                    # (1) co CLI
    if TAP.get(ten):                                                   # (2) o danh gia
        vt.add('test')
    if any(x.startswith('train (') for x in vt):
        vt.discard('train')                       # bo nhan yeu khi da co nhan xac minh
    ten_ramp = {r for r, _ in ramps}
    if not vt:                                                         # (3) cau truc
        if ten_ramp == {'ramp_base'}:
            vt.add('đo cơ sở / xác minh')
        elif glob.glob(os.path.join(RAW, ten, 'level_*')):
            vt.add('hiệu chuẩn u*')
        elif not ten_ramp:
            vt.add('đã đo, chưa vào kết quả nào' if ten.startswith('SS-')
                   else 'ngoài (benchmark / trace, không phải phép đo của bài 1)')
        else:
            vt.add('đã đo, chưa vào kết quả nào')
    return sorted(vt)


def dung_luong(p):
    try:
        return subprocess.run(['du', '-sh', p], capture_output=True, text=True).stdout.split()[0]
    except (OSError, IndexError):
        return '?'


def ngay_do(p):
    """Ngay do, lay tu dau thoi gian trong ten file manifest/collector."""
    ds = sorted({m.group(1) for f in glob.glob(os.path.join(p, '*'))
                 for m in [re.search(r'_(\d{8})_\d{6}', os.path.basename(f))] if m})
    return f'{ds[0]}..{ds[-1]}' if len(ds) > 1 else (ds[0] if ds else '')


def main():
    BO, CLI = liet_ke_bo(), co_cli()
    VT_HASH, DEP, MO_COI = khop_hash(BO)
    VT_CC, DEP_CC = khop_co_che(BO)
    for k, v in DEP_CC.items():
        DEP[k] = sorted(set(DEP.get(k, [])) | set(v))
    DEP = noi_chuoi_p3(DEP)
    MO_COI = [x for x in MO_COI if not any(x in v for v in DEP.values())]
    TAP = tap_danh_gia(BO)
    LK_O, NHAP_NHANG = lien_ket_qua_o(BO)
    KQ2 = ket_qua_bai_2(BO)
    man = {}
    for ten, ramps in sorted(BO.items()):
        man[ten] = dict(
            duong_dan=f'data/raw/{ten}',
            vai_tro=suy_vai_tro(ten, ramps, CLI, VT_HASH, VT_CC, TAP),
            tap_danh_gia=TAP.get(ten, []),
            co_ramp_base=any(r == 'ramp_base' for r, _ in ramps),
            tinh_nang=o_cua_bo(ten),
            so_run=sum(n for _, n in ramps),
            ngay_do=ngay_do(os.path.join(RAW, ten)),
            dung_luong=dung_luong(os.path.join(RAW, ten)),
            frozen_phu_thuoc=DEP.get(ten, []),
            frozen_qua_o=sorted(set(LK_O.get(ten, [])) - set(DEP.get(ten, []))),
            o_nhap_nhang=NHAP_NHANG.get(ten, []),
            ket_qua_bai_2=KQ2.get(ten, {}))
    json.dump(dict(bo_du_lieu=man, frozen_khong_khop_bo_nao=MO_COI),
              open(os.path.join(_P, 'data', 'manifest.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    L = ['# Bản đồ dữ liệu — `data/raw/`', '',
         '> Sinh tự động: `python papers/p1_du_phong/experiments/collect/sinh_manifest_du_lieu.py`.',
         '> Mọi vai trò **suy ra từ bằng chứng**, không gán tay. Chạy lại sau mỗi lần đo thêm.', '',
         '| bằng chứng | mạnh cỡ nào |', '|---|---|',
         '| khớp lại cơ chế: `fit_mechanism` trên bộ nghi vấn trùng `mechanism` đã đóng băng, từng chữ số, 7/7 dịch vụ | mạnh nhất — chứng minh **kết quả** tái lập được, không chỉ byte giống nhau |',
         '| khớp hash: `ramp_base_sha256` tính lại từ đĩa trùng fingerprint | mạnh — nhưng chỉ chứng minh byte giống nhau |',
         '| cờ CLI `--train-dir` / `--ramp-dir` trong mã nguồn | trung bình — ý định, không phải xác minh |',
         '| ô trong `p3_evaluation.csv` khớp thư mục chứa `ramp_<ô>` | yếu — ô `base` có ở **mọi** bộ, `cartsumx1` nằm ở hai bộ |',
         '| cấu trúc thư mục (chỉ `ramp_base` / có `level_*` / không ramp) | yếu nhất, chỉ dùng khi không có gì khác |', '',
         '## Không được đổi tên gì trong `data/raw/`', '',
         'Đã đo: `FP.sha256_files` băm `basename(dirname(dirname(p)))` + nội dung tệp, nên **tên ramp**',
         '(`ramp_base`, `level_150`) nằm trong hash, còn tên thư mục bộ dữ liệu thì không.', '',
         '| đổi cái gì | hậu quả |', '|---|---|',
         '| `ramp_*`, `level_*` | `train_sha256` / `ramp_base_sha256` lệch → mất khả năng xác minh bằng hash |',
         '| thư mục bộ dữ liệu | hash không đổi, nhưng chết mọi dòng CLI đã ghi trong docs/notebook và mọi đường dẫn ở đây |', '',
         '## ⚠ `train_sha256` không tái lập được — và tại sao kết quả vẫn đứng', '',
         'Hash train ghi trong **mọi** file đóng băng không khớp bất kỳ dữ liệu nào trên đĩa.',
         'Đã kiểm và **loại** sáu giả thuyết:', '',
         '| giả thuyết | kiểm bằng gì | kết quả |', '|---|---|---|',
         '| nội dung tệp đã đổi | mtime toàn bộ 16 tệp SS-TRAIN là 2026-09-21 19:50–20:58, **trước** lần đóng băng đầu (22:34) | loại |',
         '| tập tệp đã đổi | `train_files: 16` khớp đúng 16 tệp trên đĩa | loại |',
         '| hàm băm đã đổi | `git show <commit>:src/scm/feasibility_predictor.py` ở 3 commit đã ghi — `sha256_files` **giống nguyên văn** bản hiện tại | loại |',
         '| dạng đường dẫn | thử chỉ-nội-dung, đường dẫn tương đối, đường dẫn tuyệt đối | loại |',
         '| quy ước tên thư mục mức tải | thử `level_N`, `rps_N`, `N`, `load_N`, `ramp_N`, rỗng | loại |',
         '| bộ train là bộ khác | tính hash cho cả 20 bộ có `simple_metrics.csv` | loại |', '',
         '**Nguyên nhân: chưa xác định.** Nhưng điều mạnh hơn hash thì vẫn đúng: khớp lại',
         '`fit_mechanism` trên `SS-TRAIN` hiện tại cho **đúng từng chữ số** (làm tròn 5) cả `alpha`',
         'và `beta` của **7/7 dịch vụ** so với `mechanism` đã đóng băng, trên đúng 456 hàng',
         '(tải ≤ 150 req/s). Hash chứng minh byte giống nhau; phép này chứng minh **kết quả tái',
         'lập được** — đó mới là điều cần cho bài báo. Khi báo cáo, nói đúng như vậy: *cơ chế tái',
         'lập chính xác, hash train hiện không dùng được làm phép xác minh*. Đừng viện hash train.', '',
         '## Các bộ dữ liệu', '',
         '| bộ dữ liệu | vai trò | tập đánh giá | base | tính năng | run | ngày đo | dung lượng |',
         '|---|---|---|---|---|---|---|---|']
    for ten, v in man.items():
        tn = ', '.join(v['tinh_nang'][:5]) + ('…' if len(v['tinh_nang']) > 5 else '') or '—'
        L.append(f"| `{ten}` | {', '.join(v['vai_tro'])} | {', '.join(v['tap_danh_gia']) or '—'} | "
                 f"{'✓' if v['co_ramp_base'] else '—'} | {tn} | {v['so_run'] or '—'} | "
                 f"{v['ngay_do'] or '—'} | {v['dung_luong']} |")

    L += ['', '## Kết quả nào phụ thuộc bộ nào', '',
          'Khớp bằng hash nội dung và khớp lại cơ chế, **không** khớp theo tên ô — ô `base` có trong',
          'mọi bộ và `cartsumx1` nằm ở hai bộ, nên khớp tên sẽ gán sai. Các file `*_P3_*` không có',
          'fingerprint riêng; chúng trỏ về cha qua `base_frozen_file` + `base_frozen_sha256`, và',
          'chuỗi đó đã được nối vào đây. Xoá một bộ có tên dưới đây là làm các kết quả tương ứng',
          'không còn tái lập được.', '',
          '| bộ dữ liệu | file đóng băng phụ thuộc |', '|---|---|']
    for ten, v in man.items():
        if v['frozen_phu_thuoc']:
            L.append(f"| `{ten}` | {', '.join('`'+x+'`' for x in v['frozen_phu_thuoc'])} |")

    L += ['', '### Liên kết yếu hơn — khớp qua ô dự đoán', '',
          'Các bộ test còn lại không gán được bằng hash: frozen của chúng có `ramp_base_files: 0`',
          'hoặc `ramp_base` không khớp. Khớp qua **ô dự đoán** thì gán được, nhưng yếu hơn — một ô',
          'có thể nằm ở hai bộ. Mọi ô nhập nhằng đều được ghi rõ ở cột cuối.', '',
          '| bộ dữ liệu | file đóng băng (qua ô) | ô nằm ở nhiều bộ |', '|---|---|---|']
    for ten, v in man.items():
        if v['frozen_qua_o']:
            fs = ', '.join('`' + x.replace('predictions_frozen_', '') + '`' for x in v['frozen_qua_o'][:6])
            fs += f" … (+{len(v['frozen_qua_o']) - 6})" if len(v['frozen_qua_o']) > 6 else ''
            L.append(f"| `{ten}` | {fs} | {', '.join(v['o_nhap_nhang']) or '—'} |")

    if MO_COI:
        L += ['', '## File đóng băng chưa gán được về bộ nào', '',
              'Fingerprint của các file này không khớp hash của bất kỳ bộ nào hiện có, và chúng cũng',
              'không nối được về cha qua `base_frozen_file`. Cần gán tay trước khi viện dẫn chúng.', '']
        L += [f'- `{x}`' for x in MO_COI]

    L += ['', '## Dữ liệu của bài 2 — nối với CSV kết quả, không với file đóng băng', '',
          'Các file đóng băng là của **bài 1**; không file nào trỏ đến dữ liệu bài 2. Ánh xạ',
          '`he` → thư mục đọc từ bảng `HE` trong `rq6_wrapper_rcaeval.py`, không gán tay. Bộ không có',
          'cột `he` (như `alibaba`) được nối theo tiền tố tên tệp — suy luận theo tên, yếu hơn.',
          'Số trong ngoặc là số dòng kết quả rút từ bộ đó.', '',
          '| bộ dữ liệu | CSV kết quả (số dòng) |', '|---|---|']
    for ten, v in man.items():
        if v['ket_qua_bai_2']:
            top = sorted(v['ket_qua_bai_2'].items(), key=lambda kv: -kv[1])
            tong = sum(v['ket_qua_bai_2'].values())
            mo = ', '.join(f'`{k}` ({n})' for k, n in top[:4])
            mo += f" … tổng **{tong} dòng** trong {len(top)} tệp" if len(top) > 4 else ''
            L.append(f'| `{ten}` | {mo} |')

    con = [(ten, v) for ten, v in man.items()
           if not v['frozen_phu_thuoc'] and not v['frozen_qua_o'] and not v['ket_qua_bai_2']]
    L += ['', '## Bộ không có kết quả nào phụ thuộc', '',
          'Đã đo nhưng chưa vào bất kỳ kết quả nào của cả hai bài — giữ để truy vết,',
          'không dùng để công bố. Đây cũng là danh sách ứng viên nếu cần giải phóng dung lượng.', '']
    L += [f"- `{ten}` — {', '.join(v['vai_tro'])} ({v['dung_luong']})" for ten, v in con]
    L += ['', f"Tổng: **{len(con)} bộ**."]

    open(os.path.join(_P, 'data', 'DATA.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'{len(man)} bo -> data/manifest.json + data/DATA.md')
    print(f'{len(MO_COI)} file dong bang khong khop bo nao\n')
    for ten, v in man.items():
        print(f"  {ten:28s} {','.join(v['vai_tro']):22s} {str(v['tap_danh_gia'])[:38]:40s} "
              f"{len(v['frozen_phu_thuoc'])} frozen")


if __name__ == '__main__':
    main()
