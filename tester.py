#!/usr/bin/python3

from utils import *

def main():
    y = np.exp(2j * np.pi * np.arange(20) / 20)
    #y = np.random.random(1024)
    original = np.fft.fft(y)

    #dft = calculate_fft(y)
    recursive = calculate_fft(y, 'recursive')
    #iterative = calculate_fft(y, 'iterative')

    #print(np.allclose(original, dft))

    print(np.allclose(original, recursive))
    #print(np.allclose(original, iterative))



if __name__ == "__main__":
    main()
