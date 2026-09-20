# -*- coding: utf-8 -*-
"""Taxonomy hieu chinh (calibration archetypes) theo TUNG he thong dich.

Truoc day chi ton tai mot bang CALL_CHAINS duy nhat, import cung o module
level trong parser_agent.py, nen khong the tro ParserAgent sang he thu hai.
Package nay tach bang do ra theo he thong; ParserAgent nhan bang qua tham so
`call_chains`.
"""

from .sockshop import SOCKSHOP_CALL_CHAINS
from .trainticket import TRAINTICKET_CALL_CHAINS, build_trainticket_call_chains
from .sockshop_from_logs import (SOCKSHOP_LOGMINED_CALL_CHAINS,
                                  build_sockshop_call_chains_from_logs)
from .onlineboutique import ONLINEBOUTIQUE_CALL_CHAINS, build_onlineboutique_call_chains

# 'SockShop_LogMined' la MOT LUA CHON THEM, khong phai mac dinh: do chinh
# xac mine tu log (Jaccard ~0.60, xem docstring sockshop_from_logs.py)
# khong dong deu giua cac archetype nen chua thay the 'SockShop' o day.
TAXONOMIES = {
    'SockShop': SOCKSHOP_CALL_CHAINS,
    'SockShop_LogMined': SOCKSHOP_LOGMINED_CALL_CHAINS,
    'TrainTicket': TRAINTICKET_CALL_CHAINS,
    'OnlineBoutique': ONLINEBOUTIQUE_CALL_CHAINS,
}

__all__ = ['SOCKSHOP_CALL_CHAINS', 'TRAINTICKET_CALL_CHAINS',
           'build_trainticket_call_chains', 'SOCKSHOP_LOGMINED_CALL_CHAINS',
           'build_sockshop_call_chains_from_logs', 'ONLINEBOUTIQUE_CALL_CHAINS',
           'build_onlineboutique_call_chains', 'TAXONOMIES']
