"""
Pytest configuration for CI - memory cleanup only.
"""

import pytest
import gc

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup memory to prevent OOM in CI"""
    yield
    
    # Close matplotlib if imported
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
    
    # Force garbage collection
    gc.collect()