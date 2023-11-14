import numpy as np
import cv2

# Đọc ảnh DCT từ tệp ảnh DCT đã lưu
dct_image = cv2.imread('dct_image.jpg', cv2.IMREAD_GRAYSCALE)  # Chọn ảnh xám

# Kiểm tra xem ảnh đã đọc chưa
if dct_image is None:
    print("Không thể đọc ảnh DCT.")
else:
    # Thực hiện biến đổi ngược DCT 2 chiều bằng hàm idct trong OpenCV
    original_image = cv2.idct(np.float32(dct_image))

    # Chuyển đổi ma trận về kiểu dữ liệu uint8
    original_image = np.uint8(original_image)

    cv2.imwrite('idct_image.jpg', original_image)

    # Hiển thị ảnh gốc
    cv2.imshow('Original Image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()