import numpy as np


def otsu_method(img):
    # Step 1: compute the grayscale level histogram
    hist, bin = np.histogram(img.flatten(), 256, [0, 256])

    # Step 2: normalize histogram
    hist_norm = hist.astype('float32') / np.sum(hist)

    # Step 3: compute cumulative sum
    S = np.cumsum(hist_norm)

    # Step 4: compute cumulative mean
    mu = np.cumsum(hist_norm * np.arange(256))

    # Step 5: calculate the between-class variance
    mu0 = mu[-1]  # global mean value
    delta = (mu0 * S - mu) ** 2 / (S * (1 - S))

    # Step 6: find the best-fit threshold with the largest between-class variance
    threshold = np.argmax(delta)

    # Binary Equivalent Conversion
    bimg = np.array(img > threshold, dtype=np.uint8) * 255

    return bimg
