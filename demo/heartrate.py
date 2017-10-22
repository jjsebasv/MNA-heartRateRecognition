# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def main():
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     '--video',
    #     help="Path of video file"
    # )
    #
    # parser.add_argument(
    #     '--channel',
    #     help='Color channel',
    #     default="rgb",
    #     choices=['rgb', 'ycbcr']
    # )

    cap = cv2.VideoCapture('demo-video.mp4')
    # if not cap.isOpened():
    #    print("No lo pude abrir")
    #    return
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    r = np.zeros((1, length))
    g = np.zeros((1, length))
    b = np.zeros((1, length))
    y = np.zeros((1, length))
    cb = np.zeros((1, length))
    cr = np.zeros((1, length))

    k = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        # if k / length < 0.2:
        #     k += 1
        #     continue
        #
        # if k / length > 0.8:
        #     break

        if ret == True:
            r[0, k] = np.mean(frame[330:360, 610:640, 0])
            g[0, k] = np.mean(frame[330:360, 610:640, 1])
            b[0, k] = np.mean(frame[330:360, 610:640, 2])

            # r[0, k] = np.mean(frame[:, :, 0])
            # g[0, k] = np.mean(frame[:, : 1])
            # b[0, k] = np.mean(frame[:, :, 2])

            # im = rgb2ycbcr(frame[330:360, 610:640, :])
            # y[0, k] = np.mean(im[:, :, 0])
            # cb[0, k] = np.mean(im[:, :, 0])
            # cr[0, k] = np.mean(im[:, :, 0])
            print("%g%%" % (k / length * 100.))
        else:
            break
        k = k + 1

    cap.release()
    cv2.destroyAllWindows()

    n = 1024
    f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n

    r = r[0, 0:n] - np.mean(r[0, 0:n])
    g = g[0, 0:n] - np.mean(g[0, 0:n])
    b = b[0, 0:n] - np.mean(b[0, 0:n])
    # y = y[0, 0:n] - np.mean(y[0, 0:n])
    # cb = cb[0, 0:n] - np.mean(cb[0, 0:n])
    # cr = cr[0, 0:n] - np.mean(cr[0, 0:n])

    R = np.abs(np.fft.fftshift(np.fft.fft(r))) ** 2
    G = np.abs(np.fft.fftshift(np.fft.fft(g))) ** 2
    B = np.abs(np.fft.fftshift(np.fft.fft(b))) ** 2
    # Y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    # CB = np.abs(np.fft.fftshift(np.fft.fft(cb))) ** 2
    # CR = np.abs(np.fft.fftshift(np.fft.fft(cr))) ** 2
    # Y[0:30] = 0
    plt.xlabel("frecuencia [1/minuto]")

    plt.plot(60 * f, R)
    plt.xlim(0, 200)

    # plt.plot(60 * f, G)
    # plt.xlim(0, 200)
    # # plt.legend("G")

    # plt.plot(60 * f, B)
    # plt.xlim(0, 200)

    # plt.plot(60 * f, Y)
    # plt.xlim(0, 200)
    # # plt.legend("Y")

    # plt.plot(60 * f, CB)
    # plt.xlim(0, 200)

    # plt.plot(60 * f, CR)
    # plt.xlim(0, 200)

    plt.show()
    # print("Frecuencia cardíaca (Y): ", abs(f[np.argmax(Y)]) * 60, " pulsaciones por minuto")
    print("Frecuencia cardíaca (G): ", abs(f[np.argmax(G)]) * 60, " pulsaciones por minuto")


if __name__ == "__main__":
    main()
