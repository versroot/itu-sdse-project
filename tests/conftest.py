"""
Pytest configuration for CI - memory cleanup only.
"""

import pytest
import gc
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


os.makedirs("./artifacts", exist_ok=True)

# create dummy train_data_gold.csv with ALL required columns
if not os.path.exists("./artifacts/train_data_gold.csv"):
    dummy_df = pd.DataFrame({
        
        "lead_id": [1, 2, 3, 4, 5],
        "customer_code": ["C001", "C002", "C003", "C004", "C005"],
        "date_part": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],

        "purchases": [10, 20, 15, 25, 30],
        "time_spent": [100, 200, 150, 250, 300],
        "n_visits": [5, 10, 7, 12, 15],
        
        
        "customer_group": [2, 3, 4, 5, 6],
        
        
        "onboarding": [True, False, True, False, True],
        
      
        "target": [0, 1, 0, 1, 0]
    })
    dummy_df.to_csv("./artifacts/train_data_gold.csv", index=False)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup memory to prevent OOM in CI"""
    yield
    plt.close('all')
    gc.collect()
    mlruns_path = "/opt/mlruns"
    if os.path.exists(mlruns_path):
        try:
            import shutil
            shutil.rmtree(mlruns_path)
            os.makedirs(mlruns_path)
        except:
            pass