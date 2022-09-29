import numpy as np
import matplotlib.pyplot as plt
from ft import fft2d, ifft2d

def gr(P, N = -1):
    """Test fft on custom image converted to grayscale"""
    img = plt.imread(P)
    i = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    ii = fft2d(i)
    if N != -1:
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                if j+k > N:
                    ii[j, k] = 0
    iii = ifft2d(ii)
    ii[0, 0] = 0
    plt.figure("Test gray deformed" if N != -1 else "Test gray")
    plt.subplot(1, 3, 1)
    plt.imshow(i)
    plt.title("Original image")
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(ii))
    plt.title("FFT (without the first pixel)")
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(iii))
    plt.title("IFFT")
    plt.show()

if __name__ == "__main__":
    gr("img/lenna.png")
    gr("img/lenna.png", 100)
