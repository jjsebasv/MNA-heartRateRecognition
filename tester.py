#!/usr/bin/python3

from utils import *


def main():
    y = np.exp(-2j * np.pi * np.arange(1024) / 1024)
    y = np.random.random(1024)
    print(y)

    original_fft = np.fft.fft(y)

    # fft = calculate_fft(y, 'iterative')
    ifft = calculate_ifft(original_fft)
    print(ifft)

    # print(np.allclose(original_fft, fft))
    print(np.allclose(y, ifft))


if __name__ == "__main__":
    main()
