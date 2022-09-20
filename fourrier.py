# Fast Fourier Transform (FFT) and Inverse FFT (IFFT)

import numpy as np
import matplotlib.pyplot as plt

def fft(X):
    """Fast Fourier Transform (FFT)"""
    N = len(X)
    if N == 1:
        return X
    else:
        x = np.zeros(N, dtype=complex)
        for k in range(N):
            x[k] = np.sum(X*np.exp(-2j*np.pi*k*np.arange(N)/N))
        return x

def ifft(X):
    """Inverse Fast Fourier Transform (IFFT)"""
    N = len(X)
    if N == 1:
        return X
    else:
        x = np.zeros(N, dtype=complex)
        for k in range(N):
            x[k] = np.sum(X*np.exp(2j*np.pi*k*np.arange(N)/N))
        return x/N

def fft2d(X):
    """2D Fast Fourier Transform (FFT)"""
    N, M = X.shape
    x = np.array([fft(X[i]) for i in range(N)])
    x = np.array([fft(x[:,i]) for i in range(M)]).T
    return x

def ifft2d(X):
    """2D Inverse Fast Fourier Transform (FFT)"""
    N, M = X.shape
    x = np.array([ifft(X[i]) for i in range(N)])
    x = np.array([ifft(x[:,i]) for i in range(M)]).T
    return x

def d(N):
    """Test fft on 1D signal"""
    a = np.random.rand(N)
    aa = fft(a)
    aaa = ifft(aa)
    print("Original signal: ", a)
    print("FFT: ", aa)
    print("IFFT: ", aaa)
    plt.figure("Test 1D")
    plt.subplot(3, 1, 1)
    plt.stem(np.arange(N), np.abs(a))
    plt.title("Original signal")
    plt.subplot(3, 1, 2)
    plt.stem(np.arange(N), np.abs(aa))
    plt.title("FFT")
    plt.subplot(3, 1, 3)
    plt.stem(np.arange(N), np.abs(aaa))
    plt.title("IFFT")
    plt.show()
    #for i in range(N):
    #    if a[i] != aaa[i]:
    #        print("Error at position", i, "expected", a[i], "got", aaa[i])

def dd(N):
    """Test fft on 2D signal"""
    b = np.random.rand(N, N)
    bb = fft2d(b)
    bbb = ifft2d(bb)
    print("Original signal: ", b)
    print("FFT: ", bb)
    print("IFFT: ", bbb)
    plt.figure("Test 2D")
    plt.subplot(3, 1, 1)
    plt.imshow(b)
    plt.title("Original signal")
    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(bb))
    plt.title("FFT")
    plt.subplot(3, 1, 3)
    plt.imshow(np.abs(bbb))
    plt.title("IFFT")
    plt.show()
    #for i in range(N):
    #    for j in range(N):
    #        if b[i][j] != bbb[i][j]:
    #            print("Error at position", i, j, "expected", b[i][j], "got", bbb[i][j])

if __name__ == '__main__':
    N = 32
    d(N)
    dd(N)

