#!/usr/bin/python3
"""Image Compression using DTC for signals course."""


# =============================================================================
# ============================= IMPORTS =======================================
# =============================================================================
import os  # file size comparison

import cv2  # image reading showing writing etc
import matplotlib.pyplot as plt  # plotting
import numpy as np  # array / image manipulation
from scipy.fft import dctn, idctn  # DCT and IDCT

# =============================================================================


# =============================================================================
# ============================= FUNCTIONS =====================================
# =============================================================================
def process_image(image, m):
    """
    Compresses and Decompresses an image by applying 2D DCT and retaining top-left coefficients.

    Args:
        image: The image as a NumPy array.
        m: The number of top-left coefficients to retain.

    Returns:
        The decompressed image as a NumPy array.
    """
    compressed_image = np.zeros_like(image)
    decompressed_image = np.zeros_like(image)
    # NOTE: we will compress each channel separately
    for channel in range(3):
        channel_image = image[:, :, channel]
        block_height, block_width = 8, 8
        for i in range(0, image.shape[0], block_height):
            for j in range(0, image.shape[1], block_width):
                block = channel_image[i : i + block_height, j : j + block_width]
                dct_block = dctn(block)

                dct_block[m:, :] = 0
                dct_block[:, m:] = 0

                # NOTE: this is the decompression part
                idct_block = idctn(dct_block)

                # remove overflow values
                idct_block[idct_block < 0] = 0
                idct_block[idct_block > 255] = 255

                compressed_image[
                    i : i + block_height, j : j + block_width, channel
                ] = dct_block
                decompressed_image[
                    i : i + block_height, j : j + block_width, channel
                ] = idct_block
    # NOTE: saving this to file system is unnecessary made just for simulation
    cv2.imwrite(f"compressed_m_{m}.png", compressed_image)
    cv2.imwrite(f"decompressed_m_{m}.png", decompressed_image)
    return decompressed_image


def compare_sizes(m):
    """
    Compare the sizes of the original and compressed images.

    Args:
        m: The number of top-left coefficients to retain.
    """
    original_size = os.path.getsize("image1.png")
    compressed_size = os.path.getsize(f"compressed_m_{m}.png")
    print(
        f"Original Size: {original_size} byte Compressed Size: {compressed_size} byte image size ratio: {compressed_size/original_size}"
    )


def calculate_psnr(original, decompressed):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        original: The original image as a NumPy array.
        decompressed: The decompressed image as a NumPy array.

    Returns:
        The PSNR value in dB.
    """
    return cv2.PSNR(original, decompressed)


# =============================================================================


def main():
    """Do Main Function."""
    # 1. image reading
    image = cv2.imread("image1.png")
    cv2.imshow("Original", image)
    waitForEnter()

    # 2. Extract each color components
    # FIXME: this is a silly way to extract channels with reserving zeros
    red = np.zeros_like(image)
    green = np.zeros_like(image)
    blue = np.zeros_like(image)
    red[:, :, 2] = image[:, :, 2]
    green[:, :, 1] = image[:, :, 1]
    blue[:, :, 0] = image[:, :, 0]
    cv2.imshow("Red", red)
    cv2.imshow("Green", green)
    cv2.imshow("Blue", blue)
    waitForEnter()

    # 3. Loop Compress -> compare size -> decompress -> calc PSNR
    psnr_values = []
    for m in range(1, 5):
        print(f"Compressing with m={m}")
        decompressed_image = process_image(image.copy(), m)
        compare_sizes(m)
        psnr = calculate_psnr(image, decompressed_image)
        psnr_values.append(psnr)
        print(f"PSNR (m={m}): {psnr:.2f} dB")
        cv2.imshow(f"Decompressed m={m}", decompressed_image)
        waitForEnter()

    # 4. plot PSNR values
    plt.plot(range(1, 5), psnr_values, marker="o")
    plt.xlabel("m")
    plt.ylabel("PSNR")
    plt.title("PSNR vs m")
    plt.show()


def waitForEnter():
    """Wait for enter key to be pressed."""
    while True:
        if cv2.waitKey(1) == 13:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
