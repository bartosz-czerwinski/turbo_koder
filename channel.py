import math
import numpy as np

def apply_awgn_bpsk_split(systematic, parity1, parity2, snr_db):

    L = len(systematic)
    # mapowanie BPSK
    clean_s  = 2 * np.array(systematic, dtype=float) - 1.0
    clean_p1 = 2 * np.array(parity1,    dtype=float) - 1.0
    clean_p2 = 2 * np.array(parity2,    dtype=float) - 1.0

    # Kod rate
    R = 1/3.0
    # Eb/N0 w liniowej skali
    EbN0 = 10 ** (snr_db / 10.0)
    # Es/N0
    EsN0 = EbN0 * R
    # wariancja szumu
    noise_variance = 1.0 / (2.0 * EsN0)
    sigma = math.sqrt(noise_variance)

    # Dodanie szumu AWGN
    y_s  = clean_s  + np.random.normal(0.0, sigma, size=L)
    y_p1 = clean_p1 + np.random.normal(0.0, sigma, size=L)
    y_p2 = clean_p2 + np.random.normal(0.0, sigma, size=L)

    return y_s, y_p1, y_p2, noise_variance, clean_s, clean_p1, clean_p2
