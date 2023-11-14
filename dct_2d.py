import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh từ tệp hình ảnh
image = cv2.imread('og.jpg', cv2.IMREAD_GRAYSCALE)  # Chọn ảnh xám

# Kiểm tra xem ảnh đã đọc chưa
if image is None:
    print("Không thể đọc ảnh.")
else:
    # Chuyển đổi ảnh thành kiểu float32
    image_float = np.float32(image)

    # Thực hiện biến đổi DCT 2 chiều bằng hàm dct2 trong NumPy
    dct_image = cv2.dct(image_float)

    # In ra một số giá trị DCT đầu tiên (ví dụ: 8x8)
    print("Ma trận DCT đầu tiên (8x8):")
    print(dct_image[:8, :8])

    # Lưu tệp ảnh DCT nếu cần
    cv2.imwrite('dct_image.jpg', dct_image)

    # Hiển thị ảnh gốc và ảnh DCT nếu cần
    cv2.imshow('Original Image', image)
    cv2.imshow('DCT Image', dct_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

      # Lấy các hệ số DCT đầu tiên (ví dụ: 8x8)
    dct_coefficients = dct_image[:8, :8]

    # Vẽ biểu đồ 2D cho các hệ số DCT
    plt.imshow(np.log(np.abs(dct_coefficients)), cmap='gray', interpolation='none')
    plt.title("Biểu đồ 2D của các hệ số DCT")
    plt.colorbar()
    plt.show()