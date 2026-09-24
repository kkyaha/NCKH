import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'scm')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'legacy')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'model_eval')))

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
