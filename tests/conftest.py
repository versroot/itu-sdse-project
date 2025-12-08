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

# Create dummy train_data_gold.csv for import-time code
if not os.path.exists("./artifacts/train_data_gold.csv"):
    dummy_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 4, 6, 8, 10],
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