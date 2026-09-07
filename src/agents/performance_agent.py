# -*- coding: utf-8 -*-
"""
PerformanceAgent (Backward Compatibility Wrapper for CapacityAgent)
===================================================================
Lớp tương thích ngược kế thừa từ CapacityAgent.
Hỗ trợ đầy đủ toàn bộ API cũ (train_scm_models, get_metrics_for_service, ...)
đồng thời tích hợp SCM Engine thống nhất theo mô hình ReAct.
"""

import os
import sys

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_AGENT_DIR)
sys.path.insert(0, _AGENT_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))

from request_router import CALL_CHAINS
from capacity_agent import (
    CapacityAgent,
    SERVICES,
    METRICS,
    QueueingLatencyRegressor,
)

def _mape(y_true, y_pred):
    import numpy as np
    yt, yp = np.array(y_true), np.array(y_pred)
    m = yt != 0
    return np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100 if m.sum() > 0 else float('nan')


class PerformanceAgent(CapacityAgent):
    """
    PerformanceAgent tương thích ngược, vận hành trên nền CapacityAgent.
    """
    def __init__(self, data_dir: str = None, auto_train: bool = True):
        super().__init__(llm=None, data_dir=data_dir, auto_train=auto_train)
