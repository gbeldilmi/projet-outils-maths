import numpy as np
import matplotlib.pyplot as plt
from ft import fft2d, ifft2d

def rgb(P, N = -1):
    """Test fft on each channel of a custom image"""
    img = plt.imread(P)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rr = np.fft.fft2(r)
    gg = np.fft.fft2(g)
    bb = np.fft.fft2(b)
    if N != -1:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i+j > N:
                    rr[i, j] = 0
                    gg[i, j] = 0
                    bb[i, j] = 0
    rrr = np.fft.ifft2(rr)
    ggg = np.fft.ifft2(gg)
    bbb = np.fft.ifft2(bb)
    nimg = np.zeros((img.shape[0], img.shape[1], 3))
    nimg[:, :, 0] = abs(rrr)
    nimg[:, :, 1] = abs(ggg)
    nimg[:, :, 2] = abs(bbb)
    m = np.max(nimg)
    nimg /= m
    rr[0, 0] = 0
    gg[0, 0] = 0
    bb[0, 0] = 0
    plt.figure("Test rgb deformed" if N != -1 else "Test rgb")
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title("Original image")
    plt.subplot(3, 3, 4)
    plt.imshow(np.abs(rr))
    plt.title("FFT (R) (without the first pixel)")
    plt.subplot(3, 3, 5)
    plt.imshow(np.abs(gg))
    plt.title("FFT (G) (without the first pixel)")
    plt.subplot(3, 3, 6)
    plt.imshow(np.abs(bb))
    plt.title("FFT (B) (without the first pixel)")
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

if __name__ == "__main__":
    rgb("img/jl.jpg")
    rgb("img/jl.jpg", 100)
