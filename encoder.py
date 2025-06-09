import random

g = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
]

def encodedbit(g_matrix, input_bit, state):
    n = len(g_matrix)
    K = len(g_matrix[0])
    m = K - 1
    output = [0] * n
    for i in range(n):
        val = (g_matrix[i][0] * input_bit) % 2
        for j in range(1, K):
            val ^= (g_matrix[i][j] * state[j - 1]) % 2
        output[i] = val
    new_state = [input_bit] + state[: m - 1]
    return output, new_state

def turboencode(g_matrix, info_bits, terminated_flag):
    n = len(g_matrix)
    K = len(g_matrix[0])
    m = K - 1
    Linf = len(info_bits)
    # dopisanie tail bits
    if terminated_flag > 0:
        L = Linf + m
    else:
        L = Linf

    state = [0] * m
    out_sequence = []
    for i in range(L):
        if i < Linf:
            xk = info_bits[i]
        else:
            # dopisanie tail bits
            xk = sum(g_matrix[0][j] * state[j - 1] for j in range(1, K)) % 2
        xak = (g_matrix[0][0] * xk) % 2
        for j in range(1, K):
            xak ^= (g_matrix[0][j] * state[j - 1]) % 2
        outputbits, state = encodedbit(g_matrix, xak, state)
        outputbits[0] = xk
        out_sequence.extend(outputbits)
    return out_sequence

def turbo_encoder_parallel(info_bits):
    m = len(g[0]) - 1
    # pierwszy RSC
    seq1 = turboencode(g, info_bits, terminated_flag=1)
    L = len(info_bits) + m
    # rozdzielamy strumienie
    systematic = [seq1[2 * i] for i in range(L)]
    parity1    = [seq1[2 * i + 1] for i in range(L)]

    # interleaver
    interleaver = list(range(L))
    random.shuffle(interleaver)
    interleaved_info = [systematic[interleaver[i]] for i in range(L)]

    # drugi RSC
    seq2 = turboencode(g, interleaved_info, terminated_flag=1)
    parity2 = [seq2[2 * i + 1] for i in range(L)]

    return systematic, parity1, parity2, interleaver
