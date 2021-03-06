#!/usr/bin/python3


import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import recursive_fft, recursive_fft2, iterative_fft
from scipy import signal

fft_func = recursive_fft


# Filtro pasabanda obtenido de https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video',
        help="Ruta del archivo de video",
        required=True
    )

    parser.add_argument(
        '--channel',
        help='Color channel',
        default="rgb",
        choices=['rgb', 'ycbcr']
    )

    parser.add_argument(
        '--filter',
        help="Aplicar un filtro pasabanda",
        action='store_true'
    )

    parser.add_argument(
        '--lowfreq',
        help="Frecuencia minima del filtro",
        default=20,
        type=int
    )

    parser.add_argument(
        '--highfreq',
        help="Frecuencia maxima del filtro",
        default=200,
        type=int
    )

    args = parser.parse_args()

    if not args.video or not os.path.exists(args.video) or not os.path.isfile(args.video):
        print("Archivo de video no encontrado: %s" % args.video)
        return

    cap = cv2.VideoCapture(args.video)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    data = [np.zeros((1, length)), np.zeros((1, length)), np.zeros((1, length))]

    k = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            if args.channel == 'rgb':
                data[0][0, k] = np.mean(frame[200:450, 490:670, 0])
                data[1][0, k] = np.mean(frame[200:450, 490:670, 1])
                data[2][0, k] = np.mean(frame[200:450, 490:670, 2])
            elif args.channel == 'ycbcr':
                im = rgb2ycbcr(frame[200:450, 490:670, :])
                data[0][0, k] = np.mean(im[:, :, 0])
                data[1][0, k] = np.mean(im[:, :, 0])
                data[2][0, k] = np.mean(im[:, :, 0])

            # r[0, k] = np.mean(frame[:, :, 0])
            # g[0, k] = np.mean(frame[:, : 1])
            # b[0, k] = np.mean(frame[:, :, 2])

            print("%g" % (k / length * 100.), end='\r')
        else:
            break
        k = k + 1

    cap.release()
    cv2.destroyAllWindows()

    n = 1024
    f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n

    filtered_data = [0, 0, 0]
    transformed_data = [0, 0, 0]

    for i in range(0, 3):
        data[i] = data[i][0, 0:n] - np.mean(data[i][0, 0:n])

        if args.filter:
            filtered_data[i] = butter_bandpass_filter(data[i], args.lowfreq, args.highfreq, fps * 60)
            transformed_data[i] = np.abs(np.fft.fftshift(fft_func(filtered_data[i]))) ** 2
        else:
            transformed_data[i] = np.abs(np.fft.fftshift(fft_func(data[i]))) ** 2

    plt.figure(1)
    plt.clf()

    plt.plot(data[0], label='Senal original')

    if args.filter:
        plt.plot(filtered_data[0], label="Senal filtrada")

    plt.legend(loc='best')
    plt.figure(2)
    plt.clf()

    if args.channel == 'rgb':
        channels = ['R', 'G', 'B']
    elif args.channel == 'ycbcr':
        channels = ['Y', 'Cb', 'Cr']

    plt.xlabel("frecuencia [1/minuto]")
    for i in range(0, 3):
        plt.plot(60 * f, transformed_data[i], label=channels[i])

    plt.xlim(0, 200)
    plt.legend(loc='best')

    if args.channel == 'rgb':
        print("Frecuencia cardíaca: %s  pulsaciones por minuto canal R" % (abs(f[np.argmax(transformed_data[0])]) * 60))
        print("Frecuencia cardíaca: %s  pulsaciones por minuto canal G" % (abs(f[np.argmax(transformed_data[1])]) * 60))
        print("Frecuencia cardíaca: %s  pulsaciones por minuto canal B" % (abs(f[np.argmax(transformed_data[2])]) * 60))
    else:
        print("Frecuencia cardíaca: %s  pulsaciones por minuto canal Y" % (abs(f[np.argmax(transformed_data[0])]) * 60))

    plt.show()


if __name__ == "__main__":
    main()
