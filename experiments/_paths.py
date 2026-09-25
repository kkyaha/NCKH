# -*- coding: utf-8 -*-
"""Dua MOI nhom con cua experiments/ vao sys.path.

Ly do ton tai: sau khi gom experiments/ thanh thu muc con theo vai tro, 8 script van
import module o NHOM KHAC (vd model_eval/statistical_rigor can feasibility/evaluate_frozen).
Chen rieng `experiments/` vao sys.path KHONG du -- module dich nam sau mot cap nua.

Cach dung, trong file o experiments/<nhom>/:

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import _paths  # noqa: F401

Dong dau dua `experiments/` vao path (de import duoc chinh file nay); import nay lo not
phan con lai. Khong dung goi tuong doi vi cac script duoc chay TRUC TIEP nhu
`python experiments/<nhom>/<file>.py`, khong phai import nhu package.
"""
import os
import sys

_EXP = os.path.dirname(os.path.abspath(__file__))
for _d in sorted(os.listdir(_EXP)):
    _p = os.path.join(_EXP, _d)
    if os.path.isdir(_p) and not _d.startswith(('.', '__')) and _p not in sys.path:
        sys.path.insert(0, _p)
