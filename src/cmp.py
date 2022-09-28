import numpy as np
import time as bb
import matplotlib.pyplot as plt
from ft import ft, ift, fft, ifft

def t(S, D = -1):
    """Test FT on 1D signal and return the time it took"""
    N = len(S)
    r = [-1, -1]
    td = bb.time_ns()
    ss = ft(S)
    tf = bb.time_ns()
    r[0] = (tf - td) / 1000000000.
    if D != -1:
        for i in range(N):
            if i > D:
                ss[i] = 0
    print("FT: ")
    for i in range(N):
        print(i, "...", ss[i])
    td = bb.time_ns()
    sss = ift(ss)
    tf = bb.time_ns()
    r[1] = (tf - td) / 1000000000.
    print("IFT: ")
    for i in range(N):
        print(i, "...", sss[i])
    plt.subplot(2, 3, 2)
    plt.stem(np.arange(N), np.abs(ss))
    plt.title("FT")
    plt.subplot(2, 3, 3)
    plt.stem(np.arange(N), np.abs(sss))
    plt.title("IFT")
    return r

def tf(S, D = -1):
    """Test FFT on 1D signal and return the time it took"""
    N = len(S)
    r = [-1, -1]
    td = bb.time_ns()
    ss = fft(S)
    tf = bb.time_ns()
    r[0] = (tf - td) / 1000000000.
    if D != -1:
        for i in range(N):
            if i > D:
                ss[i] = 0
    print("FFT: ")
    for i in range(N):
        print(i, "...", ss[i])
    td = bb.time_ns()
    sss = ifft(ss)
    tf = bb.time_ns()
    r[1] = (tf - td) / 1000000000.
    print("IFFT: ")
    for i in range(N):
        print(i, "...", sss[i])
    plt.subplot(2, 3, 5)
    plt.stem(np.arange(N), np.abs(ss))
    plt.title("FFT")
    plt.subplot(2, 3, 6)
    plt.stem(np.arange(N), np.abs(sss))
    plt.title("IFFT")
    return r

def cmp(N, D = -1):
    """Compare the time it takes to compute the FT and the FFT"""
    s = np.random.rand(N)
    print("Original signal: ", s)
    plt.figure("Test 1D deformed" if D != -1 else "Test 1D")
    plt.subplot(2, 3, 1)
    plt.stem(np.arange(N), np.abs(s))
    plt.title("Original signal")
    ttf = t(s, D)
    ttff = tf(s, D)
    print("FT: ", ttf[0], "s\t", "IFT: ", ttf[1], "s")
    print("FFT: ", ttff[0], "s\t", "IFFT: ", ttff[1], "s")
    plt.show()

if __name__ == "__main__":
    n = 24
    d = n // 2
    cmp(n)
    cmp(n, d)
