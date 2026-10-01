# -*- coding: utf-8 -*-
"""Dua MOI nhom script cua ca hai bai vao sys.path.

Ly do ton tai: mot so script import module o NHOM KHAC (vd
model_eval/statistical_rigor can feasibility/evaluate_frozen). Cac script duoc
chay TRUC TIEP (`python <duong dan>/<file>.py`), khong import nhu package, nen
khong dung duoc goi tuong doi.

Sau khi tach hai bai, cac nhom khong con nam duoi MOT thu muc. File nay quet CA:

    <goc>/experiments/*                  (nhom chua phan loai)
    <goc>/papers/*/experiments/*         (nhom cua tung bai)

Cach dung, trong bat ky file script nao (khong phu thuoc do sau):

    _E = os.path.dirname(os.path.abspath(__file__))
    while _E != os.path.dirname(_E) and not os.path.isdir(os.path.join(_E, 'src')):
        _E = os.path.dirname(_E)
    sys.path.insert(0, os.path.join(_E, 'experiments'))
    import _paths  # noqa: F401
"""
import os
import sys

_EXP = os.path.dirname(os.path.abspath(__file__))
_GOC = os.path.dirname(_EXP)

_cay = [_EXP]
_papers = os.path.join(_GOC, 'papers')
if os.path.isdir(_papers):
    for _bai in sorted(os.listdir(_papers)):
        _e = os.path.join(_papers, _bai, 'experiments')
        if os.path.isdir(_e):
            _cay.append(_e)

for _root in _cay:
    if _root not in sys.path:
        sys.path.insert(0, _root)
    for _d in sorted(os.listdir(_root)):
        _p = os.path.join(_root, _d)
        if os.path.isdir(_p) and not _d.startswith(('.', '__')) and _p not in sys.path:
            sys.path.insert(0, _p)
