# -*- coding: utf-8 -*-
"""
SimulationAgent (Backward Compatibility Wrapper for CapacityAgent)
==================================================================
Lớp tương thích ngược kế thừa từ CapacityAgent.
Hỗ trợ đầy đủ toàn bộ API cũ (train, simulate_intervention, get_dag_summary, ...)
"""

import os
import sys

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_AGENT_DIR)
sys.path.insert(0, _AGENT_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'scm'))

from capacity_agent import (
    CapacityAgent,
    QueueingLatencyRegressor,
    SERVICES,
    METRICS,
)

class SimulationAgent(CapacityAgent):
    """
    SimulationAgent tương thích ngược, vận hành trên nền CapacityAgent.
    """
    def __init__(self, data_dir: str = None, graph_path: str = None):
        super().__init__(llm=None, data_dir=data_dir, graph_path=graph_path, auto_train=False)
