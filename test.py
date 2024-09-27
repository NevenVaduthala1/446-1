import numpy as np
from addition import add

def test_add():
    sum = add(2,2)
    assert np.allclose(sum, 4)
