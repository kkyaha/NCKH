# -*- coding: utf-8 -*-
"""Sinh lien ket render cho moi tep .puml trong thu muc nay.

Vi sao can: may nay khong co plantuml.jar. PlantUML cho phep nhung ca so do vao
URL -- deflate raw roi ma hoa base64 voi bang chu rieng. Ma hoa duoc lam HOAN
TOAN cuc bo, khong goi mang; chi khi mo lien ket moi can mang.

Chay:  python papers/p1_du_phong/docs/figures/puml/lam_lien_ket.py
Ra:    README.md trong cung thu muc, moi so do mot dong kem lien ket png + svg.

Muon render offline: tai plantuml.jar roi
    java -jar plantuml.jar -tsvg papers/p1_du_phong/docs/figures/puml/*.puml
"""
import glob
import os
import subprocess
import sys
import zlib

BANG = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'


def _ba(b1, b2, b3):
    """3 byte -> 4 ky tu, dung bang chu rieng cua PlantUML (khong phai base64 chuan)."""
    c1, c2 = b1 >> 2, ((b1 & 0x3) << 4) | (b2 >> 4)
    c3, c4 = ((b2 & 0xF) << 2) | (b3 >> 6), b3 & 0x3F
    return ''.join(BANG[c & 0x3F] for c in (c1, c2, c3, c4))


def ma_hoa(van_ban):
    """deflate raw (wbits am = khong co header zlib) roi ma hoa theo khoi 3 byte."""
    n = zlib.compressobj(9, zlib.DEFLATED, -15)
    d = n.compress(van_ban.encode('utf-8')) + n.flush()
    return ''.join(_ba(d[i], d[i + 1] if i + 1 < len(d) else 0,
                       d[i + 2] if i + 2 < len(d) else 0) for i in range(0, len(d), 3))


def kiem_cu_phap(url):
    """Hoi server PlantUML xem cu phap co loi khong -- CAN MANG.

    Vi sao can: may nay khong co JRE (`/usr/bin/java` chi la stub cua macOS), nen
    khong render offline duoc. Server tra ve HTTP 200 kem mot ANH LOI khi cu phap
    sai, nhung dat header `X-PlantUML-Diagram-Error` -- doc header la biet.
    Phep kiem nay da bat duoc loi that: `skinparam rectangle {A B}` viet mot dong
    khong hop le, 4/6 so do ban dau bi loi ma lien ket van sinh ra binh thuong.
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


def main():
    kiem = '--kiem' in sys.argv
    thu_muc = os.path.dirname(os.path.abspath(__file__))
    dong = ['# Sơ đồ PlantUML — hệ thống dự phóng khả thi', '',
            'Sinh liên kết: `python papers/p1_du_phong/docs/figures/puml/lam_lien_ket.py`.',
            'Mã hoá cục bộ (deflate + bảng base64 riêng của PlantUML), **không** gọi mạng khi sinh.',
            'Render offline: `java -jar plantuml.jar -tsvg <thư mục này>/*.puml` (máy này **chưa có JRE**).',
            'Kiểm cú pháp: thêm cờ `--kiem` (cần mạng; đọc header `X-PlantUML-Diagram-Error`).', '',
            '| sơ đồ | nội dung | xem |', '|---|---|---|']
    for p in sorted(glob.glob(os.path.join(thu_muc, '*.puml'))):
        t = open(p, encoding='utf-8').read()
        mo_ta = next((l.split('@', 1)[0].lstrip("' ").strip()
                      for l in t.split('\n') if l.startswith("' MOTA:")), '')
        mo_ta = mo_ta.replace('MOTA:', '').strip()
        m = ma_hoa(t)
        if kiem:
            loi = kiem_cu_phap(f'https://www.plantuml.com/plantuml/png/{m}')
            print(f'  {os.path.basename(p):34s} ' + (f'LOI {loi}' if loi else 'cu phap OK'))
        dong.append(f'| `{os.path.basename(p)}` | {mo_ta} | '
                    f'[png](https://www.plantuml.com/plantuml/png/{m}) · '
                    f'[svg](https://www.plantuml.com/plantuml/svg/{m}) |')
    open(os.path.join(thu_muc, 'README.md'), 'w', encoding='utf-8').write('\n'.join(dong) + '\n')
    print(f'{len(dong) - 8} so do -> {os.path.join(thu_muc, "README.md")}')


if __name__ == '__main__':
    main()
