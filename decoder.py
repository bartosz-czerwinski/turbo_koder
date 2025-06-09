import math
import numpy as np

from encoder import g

def build_trellis_and_parity(g_matrix):
    K = len(g_matrix[0])
    m = K - 1
    nstates = 2 ** m

    trellismat = np.zeros((nstates, nstates), dtype=int)
    paritymat = np.zeros((nstates, nstates), dtype=int)

    for s in range(nstates):
        bins = [(s >> bit) & 1 for bit in reversed(range(m))]
        xk = 0
        xak = (g_matrix[0][0] * xk) % 2
        for j in range(1, K):
            xak ^= (g_matrix[0][j] * bins[j - 1]) % 2
        nstate_bits = [xak] + bins[: m - 1]
        s0 = 0
        for bit in nstate_bits:
            s0 = (s0 << 1) | bit
        v = (g_matrix[1][0] * xak) % 2
        for j in range(1, K):
            v ^= (g_matrix[1][j] * bins[j - 1]) % 2
        trellismat[s, s0] = -1
        paritymat[s, s0] = 2 * v - 1

        xk = 1
        xak = (g_matrix[0][0] * xk) % 2
        for j in range(1, K):
            xak ^= (g_matrix[0][j] * bins[j - 1]) % 2
        nstate_bits = [xak] + bins[: m - 1]
        s1 = 0
        for bit in nstate_bits:
            s1 = (s1 << 1) | bit
        v = (g_matrix[1][0] * xak) % 2
        for j in range(1, K):
            v ^= (g_matrix[1][j] * bins[j - 1]) % 2
        trellismat[s, s1] = +1
        paritymat[s, s1] = 2 * v - 1

    return trellismat, paritymat

def bcjr_decode(
    trellismat: np.ndarray,
    paritymat: np.ndarray,
    y_sys: np.ndarray,
    y_par: np.ndarray,
    L_a: np.ndarray,
    L_c: float,
    decoder_index: int
):
    L = y_sys.shape[0]
    nstates = trellismat.shape[0]

    La_expanded = L_a.reshape((L, 1, 1))
    sys_term   = (L_c * y_sys).reshape((L, 1, 1))
    par_term   = (L_c * y_par).reshape((L, 1, 1))

    trelli = trellismat[np.newaxis, :, :]
    parity = paritymat[np.newaxis, :, :]

    metric = trelli * (La_expanded + sys_term) + parity * par_term

    # ograniczona wartość metryki, żeby uniknąć zbyt dużych wartości i wykrzaczenia programu:
    metric_clipped = np.clip(metric, -100.0, 100.0)

    gamma = np.exp(0.5 * metric_clipped)

    mask_zero = (trellismat == 0)
    gamma[:, mask_zero] = 0.0

    alpha = np.zeros((nstates, L + 1), dtype=float)
    alpha[:, 0] = 0.0
    alpha[0, 0] = 1.0

    for k in range(L):
        alpha[:, k + 1] = gamma[k].T.dot(alpha[:, k])
        s = alpha[:, k + 1].sum()
        if s != 0.0 and not np.isnan(s):
            alpha[:, k + 1] /= s

    beta = np.zeros((nstates, L + 1), dtype=float)
    if decoder_index == 1:
        beta[:, L] = 0.0
        beta[0, L] = 1.0
    else:
        beta[:, L] = 1.0 / float(nstates)

    for k in range(L - 1, -1, -1):
        beta[:, k] = gamma[k].dot(beta[:, k + 1])
        s = beta[:, k].sum()
        if s != 0.0 and not np.isnan(s):
            beta[:, k] /= s

    L_out = np.zeros(L, dtype=float)
    L_e   = np.zeros(L, dtype=float)

    mask_pos = (trellismat == +1)
    mask_neg = (trellismat == -1)

    for k in range(L):
        M = alpha[:, k].reshape((nstates, 1)) * gamma[k] * beta[:, k + 1].reshape((1, nstates))

        num = M[mask_pos].sum()
        den = M[mask_neg].sum()

        if num <= 0.0 or den <= 0.0 or np.isnan(num) or np.isnan(den):
            L_out[k] = 0.0
        else:
            L_out[k] = math.log(num / den)

        L_e[k] = L_out[k] - L_a[k] - L_c * y_sys[k]

    return L_out, L_e

def turbo_decoder(y_s, y_p1, y_p2, interleaver, noise_variance, n_iter=5, m=4):

    L_ext = y_s.shape[0]
    L_info = L_ext - m

    trellismat, paritymat = build_trellis_and_parity(g)
    nstates = trellismat.shape[0]
    L_c = 2.0 / noise_variance

    La1 = np.zeros(L_ext, dtype=float)
    deinterleaver = np.argsort(interleaver)

    for it in range(n_iter):
        # dekoder 1
        L_out1, L_ext1 = bcjr_decode(
            trellismat, paritymat,
            y_s, y_p1,
            La1, L_c,
            decoder_index=1
        )
        La2 = L_ext1[interleaver]
        y_s_int = y_s[interleaver]

        # dekoder 2
        L_out2_int, L_ext2_int = bcjr_decode(
            trellismat, paritymat,
            y_s_int, y_p2,
            La2, L_c,
            decoder_index=2
        )
        La1 = L_ext2_int[deinterleaver]

    # finalny wektor LLR
    LLR_ext = L_out2_int[deinterleaver]
    x_ext   = (LLR_ext > 0).astype(int)

    # odrzucenie tail bits
    LLR_final = LLR_ext[:L_info]
    x_hat     = x_ext[:L_info]

    return LLR_final, x_hat
