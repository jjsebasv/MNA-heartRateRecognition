import numpy as np
import time


def iterative_fft(y):
    N = y.shape[0]
    N_min = min(N, 32)

    # Hace dft de a N_min todos juntos
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, y.reshape((N_min, -1)))

    # Replica lo recursivo
    while X.shape[0] < N:
        rows = int(X.shape[0])
        cols = int(X.shape[1])
        X_even = X[:, :(int(cols / 2))]
        X_odd = X[:, (int(cols / 2)):]
        factor = np.exp(-1j * np.pi * np.arange(rows) / rows)[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])

    return X.ravel()


def recursive_fft(y):
    N = len(y)
    if N <= 32: return calculate_dft(y)
    even = recursive_fft(y[0::2])
    odd = recursive_fft(y[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(0, int(N / 2))]
    ans = [even[k] + T[k] for k in range(0, int(N / 2))] + [even[k] - T[k] for k in range(0, int(N / 2))]
    return ans


def calculate_fft(y, method=''):
    N = len(y)

    if method == 'iterative' and np.log2(N) % 1 <= 0:
        print("*** Iterative FFT calculation ***")
        ans = iterative_fft(y)
    elif method == 'recursive' and np.log2(N) % 1 <= 0:
        print("*** Recursive FFT calculation ***")
        ans = recursive_fft(y)
    else:
        print("*** DFT calculation ***")
        ans = calculate_dft(y)
    return ans


def calculate_dft(y):
    N = y.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, y)


def get_time():
    return time.time()
