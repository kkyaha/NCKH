import sys
import os

_GOC = os.path.dirname(os.path.abspath(__file__))   # leo len den thu muc chua src/
while _GOC != os.path.dirname(_GOC) and not os.path.isdir(os.path.join(_GOC, 'src')):
    _GOC = os.path.dirname(_GOC)
sys.path.append(os.path.join(_GOC, 'src', 'scm'))
sys.path.insert(0, os.path.join(_GOC, 'experiments'))
import _paths  # noqa: E402,F401  -- dua MOI nhom script cua ca hai bai vao sys.path

from data_processor import load_normal_data
from future_rca import OODGuard

def test_ood_guard():
    import pandas as pd
    import numpy as np
    
    # Mock data
    train_data = pd.DataFrame({
        'front-end_workload': np.random.normal(50, 5, 1000)
    })
    
    guard = OODGuard(train_data)
    
    # In-distribution
    conf, _, _, _ = guard.check({'front-end_workload': 50})
    assert conf == 'high'
    
    # Extrapolation
    conf, _, _, _ = guard.check({'front-end_workload': 200})
    assert conf == 'very_low'

if __name__ == '__main__':
    test_ood_guard()
    print("All tests passed.")
