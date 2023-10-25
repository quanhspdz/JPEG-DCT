import numpy as np
import cv2
import scipy.fftpack as fft
from scipy.fftpack import idct
import matplotlib.pyplot as plt

# Hàm tích chập 1D giữa hai mảng
def convolve1D(data, kernel):
    return [sum(data[i:i+len(kernel)] * kernel) for i in range(len(data) - len(kernel) + 1)]

# Hàm biến đổi DCT 1D
def dct1D(data):
    N = len(data)
    dct = np.zeros(N)
    for k in range(N):
        sum_val = 0.0
        for n in range(N):
            sum_val += data[n] * np.cos((np.pi * k / N) * (n + 0.5))
        dct[k] = sum_val
    return dct

# Hàm biến đổi DCT 2D cho một khối 8x8
def dct2D(block):
    # Biến đổi DCT trên các hàng
    row_dct = [dct1D(row) for row in block]
    
    # Biến đổi DCT trên các cột của ma trận đã được biến đổi DCT từ trước
    col_dct = [dct1D(np.array(row_dct).T) for row_dct in block]

    return np.array(col_dct)

# Hàm lấy một khối 8x8 từ ảnh
def get_image_block(image, i, j):
    return image[i:i+8, j:j+8]

# Hàm nén ảnh bằng DCT
def compress_image(image):
    height, width = image.shape
    compressed_image = np.zeros((height, width))

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = get_image_block(image, i, j)
            dct_block = dct2D(block)
            compressed_image[i:i+8, j:j+8] = dct_block

    return compressed_image

# Hàm giải nén ảnh bằng DCT
def decompress_image(compressed_image):
    height, width = compressed_image.shape
    decompressed_image = np.zeros((height, width))

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = get_image_block(compressed_image, i, j)
            idct_block = np.round(np.real(fft.idct(fft.idct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')))
            decompressed_image[i:i+8, j:j+8] = idct_block

    return decompressed_image

# Đọc ảnh gốc
# Đọc ảnh từ tệp hình ảnh
original_image = cv2.imread('og.jpg', cv2.IMREAD_GRAYSCALE)  # Chọn ảnh xám

# Kiểm tra xem ảnh đã đọc chưa
if original_image is None:
    print("Không thể đọc ảnh.")
else:
    # Lấy kích thước ảnh
    height, width = original_image.shape[:2]

    print(f'Kích thước ảnh: {width}x{height}')

    # Nén ảnh
    compressed_image = compress_image(original_image)

    # Lưu tệp ảnh DCT nếu cần
    cv2.imwrite('dct_image.jpg', compressed_image)

    # In ra ma trận điểm ảnh ban đầu
    print("Ma trận đầu tiên ảnh ban đầu (8x8):")
    print(original_image[:8, :8])

    # In ra một số giá trị DCT đầu tiên (ví dụ: 8x8)
    print("Ma trận DCT đầu tiên (8x8):")
    print(compressed_image[:8, :8])

    # Giải nén ảnh
    decompressed_image = decompress_image(compressed_image)

    # Lưu tệp ảnh giải nén
    cv2.imwrite('decompressed_image.jpg', decompressed_image)

    # # Hiển thị ảnh gốc và ảnh giải nén
    # cv2.imshow('Compressed Image', compressed_image)
    # cv2.imshow('Decompressed Image', decompressed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Vẽ biểu đồ 2D cho các hệ số DCT
    plt.imshow(np.log(np.abs(compressed_image)), cmap='gray', interpolation='none')
    plt.title("Biểu đồ 2D của các hệ số DCT")
    plt.colorbar()
    plt.show()