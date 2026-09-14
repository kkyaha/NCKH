# -*- coding: utf-8 -*-
"""Taxonomy hieu chinh (calibration archetypes) theo TUNG he thong dich.

Truoc day chi ton tai mot bang CALL_CHAINS duy nhat, import cung o module
level trong parser_agent.py, nen khong the tro ParserAgent sang he thu hai.
Package nay tach bang do ra theo he thong; ParserAgent nhan bang qua tham so
`call_chains`.
"""

from .sockshop import SOCKSHOP_CALL_CHAINS
from .trainticket import TRAINTICKET_CALL_CHAINS, build_trainticket_call_chains

TAXONOMIES = {
    'SockShop': SOCKSHOP_CALL_CHAINS,
    'TrainTicket': TRAINTICKET_CALL_CHAINS,
}

__all__ = ['SOCKSHOP_CALL_CHAINS', 'TRAINTICKET_CALL_CHAINS',
           'build_trainticket_call_chains', 'TAXONOMIES']
