import numpy as np

def ft(X):
    """Fourier Transform (FT)"""
    N = len(X)
    if N == 1:
        return X
    else:
        x = np.zeros(N, dtype=complex)
        for k in range(N):
            x[k] = np.sum(X*np.exp(-2j*np.pi*k*np.arange(N)/N))
        return x

def ift(X):
    """Inverse Fourier Transform (IFT)"""
    N = len(X)
    if N == 1:
        return X
    else:
        x = np.zeros(N, dtype=complex)
        for k in range(N):
            x[k] = np.sum(X*np.exp(2j*np.pi*k*np.arange(N)/N))
        return x/N

def fft(X):
    """Fast Fourier Transform (FFT)"""
    N = len(X)
    if N == 1:
        return X
    else:
        x = np.zeros(N, dtype=complex)
        xe = fft(X[::2])
        xo = fft(X[1::2])
        for k in range(N//2):
            p = xe[k]
            q = np.exp(-2j*np.pi*k/N) * xo[k]
            x[k] = p+q
            x[k+(N//2)] = p-q
        return x

def ifft(X):
    """Inverse Fast Fourier Transform (IFFT)"""
    N = len(X)
    if N == 1:
        return X
    else:
        x = np.zeros(N, dtype=complex)
        xe = ifft(X[::2])
        xo = ifft(X[1::2])
        for k in range(N//2):
            p = xe[k]
            q = np.exp(2j*np.pi*k/N) * xo[k]
            x[k] = p+q
            x[k+(N//2)] = p-q
        return x/N

def ft2d(X):
    """2D Fourier Transform (FT)"""
    N, M = X.shape
    x = np.array([ft(X[i]) for i in range(N)])
    x = np.array([ft(x[:,i]) for i in range(M)]).T
    return x

def ift2d(X):
    """2D Inverse Fourier Transform (FT)"""
    N, M = X.shape
    x = np.array([ift(X[i]) for i in range(N)])
    x = np.array([ift(x[:,i]) for i in range(M)]).T
    return x

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
