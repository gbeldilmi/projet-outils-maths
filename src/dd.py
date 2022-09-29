import numpy as np
import matplotlib.pyplot as plt
from ft import fft2d, ifft2d

def dd(N, D = -1):
    """Test fft on a random 2D signal"""
    s = np.random.rand(N, N)
    ss = fft2d(s)
    if D != -1:
        for i in range(N):
            for j in range(N):
                if i+j > D:
                    ss[i, j] = 0
    sss = ifft2d(ss)
    ss[0, 0] = 0
    plt.figure("Test 2D deformed" if D != -1 else "Test 2D")
    plt.subplot(1, 3, 1)
    plt.imshow(s)
    plt.title("Original signal")
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(ss))
    plt.title("FFT (without the first pixel)")
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(sss))
    plt.title("IFFT")
    plt.show()

if __name__ == "__main__":
    n = 8
    d = 2
    dd(n)
    dd(n, d)
