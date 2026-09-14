# -*- coding: utf-8 -*-
"""Taxonomy SockShop -- tai xuat tu request_router de moi taxonomy nam cung cho.

Khong dinh nghia lai bang o day: request_router.SOCKSHOP_CALL_CHAINS van la
nguon su that duy nhat, tranh hai ban sao lech nhau.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from request_router import SOCKSHOP_CALL_CHAINS  # noqa: E402

__all__ = ['SOCKSHOP_CALL_CHAINS']
