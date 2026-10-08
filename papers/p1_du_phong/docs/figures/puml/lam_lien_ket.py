# -*- coding: utf-8 -*-
"""Sinh lien ket render cho moi tep .puml trong thu muc nay.

Vi sao can: may nay khong co JRE (`/usr/bin/java` chi la stub cua macOS), nen
khong render offline duoc. PlantUML cho phep nhung ca so do vao URL -- deflate raw
roi ma hoa base64 voi bang chu rieng. Ma hoa lam HOAN TOAN cuc bo, khong goi mang;
chi khi mo lien ket (hoac khi dung co --kiem) moi can mang.

NGUYEN TAC CHO MOI SO DO: mot y chinh, doc duoc tu cuoi phong. Khong p-value,
khong ten co che, khong so node tren hinh -- nhung thu do de tra loi khi BI HOI,
cho vao HE_THONG.md. Tren hinh chi giu con so nao TU NO la lap luan.
Da do: ban dau 5/7 so do "qua nang" (03a: 45 con so, 595 tu) -- khong dung duoc.

Chay:  python papers/p1_du_phong/docs/figures/puml/lam_lien_ket.py [--kiem]
Ra:    README.md trong cung thu muc.

Render offline (can JRE): java -jar plantuml.jar -tsvg <thu muc nay>/*.puml
"""
import glob
import os
import subprocess
import sys
import zlib

BANG = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'


def _ba(b1, b2, b3):
    """3 byte -> 4 ky tu, bang chu RIENG cua PlantUML (khong phai base64 chuan)."""
    c1, c2 = b1 >> 2, ((b1 & 0x3) << 4) | (b2 >> 4)
    c3, c4 = ((b2 & 0xF) << 2) | (b3 >> 6), b3 & 0x3F
    return ''.join(BANG[c & 0x3F] for c in (c1, c2, c3, c4))


def ma_hoa(van_ban):
    """deflate raw (wbits am = khong header zlib) roi ma hoa theo khoi 3 byte."""
    n = zlib.compressobj(9, zlib.DEFLATED, -15)
    d = n.compress(van_ban.encode('utf-8')) + n.flush()
    return ''.join(_ba(d[i], d[i + 1] if i + 1 < len(d) else 0,
                       d[i + 2] if i + 2 < len(d) else 0) for i in range(0, len(d), 3))


def kiem_cu_phap(url):
    """Hoi server PlantUML xem cu phap co loi khong -- CAN MANG.

    Server tra HTTP 200 kem mot ANH LOI khi cu phap sai, nhung dat header
    `X-PlantUML-Diagram-Error`. Phep kiem nay da bat duoc loi that:
    `skinparam rectangle {A B}` viet mot dong khong hop le, 4/6 so do ban dau bi
    loi ma lien ket van sinh ra binh thuong.
    """
    try:
        r = subprocess.run(['curl', '-sS', '-m', '30', '-D', '-', '-o', os.devnull, url],
                           capture_output=True, text=True, timeout=40)
    except (OSError, subprocess.TimeoutExpired):
        return None                                  # khong co mang -> khong ket luan
    h = {k.strip().lower(): v.strip() for k, v in
         (l.split(':', 1) for l in r.stdout.split('\n') if ':' in l)}
    loi = h.get('x-plantuml-diagram-error')
    return None if not loi else f"dong {h.get('x-plantuml-diagram-error-line', '?')}: {loi}"


def mo_ta_cua(t):
    """Dong `' MOTA: ...` o dau tep."""
    for l in t.split('\n'):
        if l.startswith("' MOTA:"):
            return l.split('MOTA:', 1)[1].strip()
    return ''


def hang_bang(tep, kiem):
    out = []
    for p in tep:
        t = open(p, encoding='utf-8').read()
        m = ma_hoa(t)
        if kiem:
            loi = kiem_cu_phap(f'https://www.plantuml.com/plantuml/png/{m}')
            print(f'  {os.path.basename(p):36s} ' + (f'LOI {loi}' if loi else 'cu phap OK'))
        out.append(f'| `{os.path.basename(p)}` | {mo_ta_cua(t)} | '
                   f'[png](https://www.plantuml.com/plantuml/png/{m}) · '
                   f'[svg](https://www.plantuml.com/plantuml/svg/{m}) |')
    return out


def main():
    kiem = '--kiem' in sys.argv
    thu_muc = os.path.dirname(os.path.abspath(__file__))
    tep = sorted(glob.glob(os.path.join(thu_muc, '*.puml')))
    dong = ['# Sơ đồ PlantUML — hệ thống dự phóng khả thi', '',
            '> Sinh liên kết: `python papers/p1_du_phong/docs/figures/puml/lam_lien_ket.py`.',
            '> Mã hoá cục bộ (deflate + bảng base64 riêng của PlantUML), **không** gọi mạng khi sinh.',
            '> Kiểm cú pháp: thêm cờ `--kiem` (cần mạng; đọc header `X-PlantUML-Diagram-Error`).',
            '> Render offline cần JRE — máy này **chưa có**:',
            '> `java -jar plantuml.jar -tsvg <thư mục này>/*.puml`.', '',
            'Mỗi hình mang **một** ý và đọc được từ cuối phòng. Số liệu đầy đủ, p-value và tên',
            'cơ chế nằm ở `docs/HE_THONG.md` và các notebook — để trả lời khi bị hỏi, không lên hình.',
            '', '| sơ đồ | nội dung | xem |', '|---|---|---|']
    dong += hang_bang(tep, kiem)
    open(os.path.join(thu_muc, 'README.md'), 'w', encoding='utf-8').write('\n'.join(dong) + '\n')
    print(f'{len(tep)} so do -> README.md')


if __name__ == '__main__':
    main()
