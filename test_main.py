import numpy as np
from main import *

def test_konvolucija_identity():
    img = np.array([[1, 2], [3, 4]], dtype=np.float32)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    result = konvolucija(img, kernel)

    assert np.allclose(result, img)

def test_horizon_flat():
    img = np.zeros((100, 100), dtype=np.float32)
    img[50:] = 1

    angle = oceni_orientacijo_horizonta(img)

    assert abs(abs(angle) - 90) < 10