import numpy as np
import cv2 as cv
from pathlib import Path
BASE = Path(__file__).parent

def konvolucija(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    if len(slika.shape) == 2:
        visina, sirina = slika.shape
        kanali = 1
        slika = slika[:, :, np.newaxis]
    else:
        visina, sirina, kanali = slika.shape

    kh, kw = jedro.shape
    pad_h = kh // 2
    pad_w = kw // 2

    nova_slika = np.zeros_like(slika)

    for i in range(visina):
        for j in range(sirina):
            for k in range(kanali):
                vsota = 0
                for m in range(-pad_h, pad_h + 1):
                    for n in range(-pad_w, pad_w + 1):
                        ii = max(0, min(i + m, visina - 1))
                        jj = max(0, min(j + n, sirina - 1))
                        vsota += slika[ii, jj, k] * jedro[m + pad_h, n + pad_w]

                nova_slika[i, j, k] = vsota

    if kanali == 1:
        return nova_slika[:, :, 0]
    return nova_slika

def sobel_vertikalno(slika: np.ndarray, max_gradient: np.float32, barva: tuple) -> np.ndarray:
    if len(slika.shape) == 3:
        slika_gray = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        slika_gray = slika.copy()

    jedro_sober = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    gradient = konvolucija(slika_gray, jedro_sober)
    gradient = np.abs(gradient)

    nova_slika = cv.cvtColor(slika_gray, cv.COLOR_GRAY2BGR)

    visina, sirina = gradient.shape

    for i in range(visina):
        for j in range(sirina):
            if gradient[i, j] > max_gradient:
                nova_slika[i, j] = barva

    return nova_slika

def poisci_koticke_rotiranih_kvadratov(slika: np.ndarray) -> np.ndarray:
    if len(slika.shape) == 3:
        slika_gray = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        slika_gray = slika.copy()

    slika_gray = slika_gray.astype(np.float32)
    slika_gray = cv.GaussianBlur(slika_gray, (5,5), 0)

    gx = cv.Sobel(slika_gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(slika_gray, cv.CV_32F, 0, 1, ksize=3)

    g45 = (gx + gy) # /
    g135 = (gx - gy) # \

    Z = np.abs(np.minimum(g45, 0) * np.maximum(g135, 0))
    D = np.abs(np.maximum(g45, 0) * np.maximum(g135, 0))
    L = np.abs(np.minimum(g45, 0) * np.minimum(g135, 0))
    S = np.abs(np.maximum(g45, 0) * np.minimum(g135, 0))

    def norm(x):
        return cv.normalize(x, None, 0, 1, cv.NORM_MINMAX)

    Z = norm(Z)
    D = norm(D)
    L = norm(L)
    S = norm(S)

    visina, sirina = slika_gray.shape
    rezultat = np.zeros((visina, sirina, 4), dtype=np.float32)
    rezultat[:, :, 0] = Z
    rezultat[:, :, 1] = D
    rezultat[:, :, 2] = L
    rezultat[:, :, 3] = S

    return rezultat

def poisci_znak_a(slika: np.ndarray) -> np.ndarray:
    if len(slika.shape) == 3:
        slika_gray = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        slika_gray = slika.copy()

    slika_gray = slika_gray.astype(np.float32)
    jedro_A = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1]
        # [1, 1, 1],
        # [1, 0, 1],
        # [1, 1, 1],
        # [1, 0, 1],
        # [1, 0, 1]
    ])
    nova_slika = konvolucija(slika_gray, jedro_A)
    nova_slika = np.abs(nova_slika)
    return nova_slika

def oceni_orientacijo_horizonta(slika: np.ndarray) -> float:
    if len(slika.shape) == 3:
        slika_gray = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        slika_gray = slika.copy()

    slika_gray = slika_gray.astype(np.float32)
    slika_gray = cv.GaussianBlur(slika_gray, (5, 5), 0)

    gx = cv.Sobel(slika_gray, cv.CV_32F, 1, 0, ksize=3)  # по x
    gy = cv.Sobel(slika_gray, cv.CV_32F, 0, 1, ksize=3)  # по y

    theta = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    angles_deg = np.degrees(theta)
    angles_deg = (angles_deg + 90) % 180 - 90

    hist, bins = np.histogram(angles_deg, bins=180, range=(-90, 90), weights=magnitude)

    horizon_angle = bins[np.argmax(hist)]

    return horizon_angle

# if __name__ == "__main__":
#     slika = cv.imread(str(BASE / "letter.png")).astype(np.float32) / 255
#     filtr_slika = poisci_znak_a(slika)
#     cv.imshow("Original", slika)
#     cv.imshow("filtr", filtr_slika)
#     cv.waitKey(0)
#     cv.destroyAllWindows()