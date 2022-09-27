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

def d(N, deform = False):
    """Test fft on 1D signal"""
    a = np.random.rand(N)
    aa = fft(a)
    aaa = ifft(aa)
    print("Original signal: ", a)
    print("FFT: ", aa)
    if deform:
        for i in range(N):
            if i >= N//2:
                aa[i] = 0
    print("IFFT: ", aaa)
    for i in range(N):
        if a[i] != aaa[i]:
            print("Error at position", i, "expected", a[i], "got", aaa[i])
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

def dd(N, deform = False):
    """Test fft on 2D signal"""
    b = np.random.rand(N, N)
    bb = fft2d(b)
    bbb = ifft2d(bb)
    print("Original signal: ", b)
    print("FFT: ", bb)
    if deform:
        for i in range(N):
            for j in range(N):
                if i+j > N-1:
                    bb[i, j] = 0
    print("IFFT: ", bbb)
    for i in range(N):
        for j in range(N):
            if b[i][j] != bbb[i][j]:
                print("Error at position", i, j, "expected", b[i][j], "got", bbb[i][j])
    bb[0, 0] = 0
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

def img(path, N = -1):
    """Test fft on custom image converted to grayscale"""
    img = plt.imread(path)
    i = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    ii = fft2d(i)
    iii = ifft2d(ii)
    print("Original image: ", i)
    print("FFT: ", ii)
    if N != -1:
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                if j+k > N:
                    ii[j, k] = 0
    print("IFFT: ", iii)
    ii[0, 0] = 0
    plt.figure("Test gray")
    plt.subplot(3, 1, 1)
    plt.imshow(i)
    plt.title("Original image")
    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(ii))
    plt.title("FFT")
    plt.subplot(3, 1, 3)
    plt.imshow(np.abs(iii))
    plt.title("IFFT")
    plt.show()

def rgb(path, N = -1):
    """Test fft on each channel of a custom image"""
    img = plt.imread(path)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rr = fft2d(r)
    gg = fft2d(g)
    bb = fft2d(b)
    rrr = ifft2d(rr)
    ggg = ifft2d(gg)
    bbb = ifft2d(bb)
    print("Original image: ", img)
    print("FFT: ", rr, gg, bb)
    if N != -1:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i+j > N:
                    rr[i, j] = 0
                    gg[i, j] = 0
                    bb[i, j] = 0
    print("IFFT (R): ", rrr)
    print("IFFT (G): ", ggg)
    print("IFFT (B): ", bbb)
    nimg = np.zeros((img.shape[0], img.shape[1], 3))
    nimg[:, :, 0] = rrr
    nimg[:, :, 1] = ggg
    nimg[:, :, 2] = bbb
    rr[0, 0] = 0
    gg[0, 0] = 0
    bb[0, 0] = 0
    plt.figure("Test rgb")
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(3, 3, 4)
    plt.imshow(np.abs(rr))
    plt.title("FFT (R)")
    plt.subplot(3, 3, 5)
    plt.imshow(np.abs(gg))
    plt.title("FFT (G)")
    plt.subplot(3, 3, 6)
    plt.imshow(np.abs(bb))
    plt.title("FFT (B)")
    plt.subplot(3, 3, 7)
    plt.imshow(np.abs(rrr))
    plt.title("IFFT (R)")
    plt.subplot(3, 3, 8)
    plt.imshow(np.abs(ggg))
    plt.title("IFFT (G)")
    plt.subplot(3, 3, 9)
    plt.imshow(np.abs(bbb))
    plt.title("IFFT (B)")
    plt.subplot(3, 3, 3)
    plt.imshow(nimg)
    plt.title("Reconstructed image")
    plt.show()

if __name__ == '__main__':
    N = 12
    d(N)
    d(N, True)
    dd(N)
    dd(N, True)
    img("lenna.png")
    img("lenna.png", N)
    rgb("lenna.png")
    rgb("lenna.png", N)
