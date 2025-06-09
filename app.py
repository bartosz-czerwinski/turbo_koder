# app.p
import random
from flask import Flask, render_template, request

from encoder import turbo_encoder_parallel
from decoder import turbo_decoder
from channel import apply_awgn_bpsk_split

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    - GET: wyświetla formularz.
    - POST: generuje losowe bity, koduje, dodaje AWGN, dekoduje, liczy BER.
    """
    # Inicjalizacja zmiennych
    bits_list = None # lista bitów wejściowych
    systematic = None
    parity1 = None
    parity2 = None
    interleaver = None

    y_s = y_p1 = y_p2 = None # zaszumione ciągi BPSK
    clean_s = clean_p1 = clean_p2 = None # czyste BPSK
    noise_variance = None

    LLR_final = None
    x_hat = None
    ber = None

    snr_db = None
    n_iter = None

    if request.method == 'POST':
        # pobranie parametrów z formularza
        try:
            length = int(request.form.get('bit_length', 0))
        except ValueError:
            length = 0

        try:
            snr_db = float(request.form.get('snr_db', 0.0))
        except ValueError:
            snr_db = 0.0

        try:
            n_iter = int(request.form.get('n_iter', 5))
        except ValueError:
            n_iter = 5

        if length > 0:
            # losowanie bitów
            bits_list = [random.randint(0, 1) for _ in range(length)]

            # kodowanie
            systematic, parity1, parity2, interleaver = turbo_encoder_parallel(bits_list)

            # modulacja BPSK i dodanie szumu AWGN
            y_s, y_p1, y_p2, noise_variance, clean_s, clean_p1, clean_p2 = (
                apply_awgn_bpsk_split(systematic, parity1, parity2, snr_db)
            )

            # dekodowanie
            LLR_final, x_hat = turbo_decoder(y_s, y_p1, y_p2, interleaver, noise_variance, n_iter=n_iter)

            # obliczenie BER
            errors = sum(b != xh for b, xh in zip(bits_list, x_hat))
            ber = errors / float(length)
        else:
            bits_list = []
            snr_db = None
            n_iter = None

    # Renderowanie szablonu
    return render_template(
        'index.html',
        bits=bits_list,
        systematic=systematic,
        parity1=parity1,
        parity2=parity2,
        interleaver=interleaver,
        clean_s=clean_s,
        clean_p1=clean_p1,
        clean_p2=clean_p2,
        y_s=y_s,
        y_p1=y_p1,
        y_p2=y_p2,
        noise_variance=noise_variance,
        LLR_final=LLR_final,
        x_hat=x_hat,
        ber=ber,
        snr_db=snr_db,
        n_iter=n_iter
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6050, debug=True)
