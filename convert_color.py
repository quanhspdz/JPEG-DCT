import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the grayscale image
gray_image = cv2.imread('og.jpg', cv2.IMREAD_GRAYSCALE)

# Create an empty RGB image
height, width = gray_image.shape
rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

# Copy the grayscale values into all three channels
rgb_image[:, :, 0] = gray_image
rgb_image[:, :, 1] = gray_image
rgb_image[:, :, 2] = gray_image

# Display the original grayscale and the converted RGB image
plt.subplot(121), plt.imshow(gray_image, cmap='gray'), plt.axis('off'), plt.title('Grayscale Image', size=10)
plt.subplot(122), plt.imshow(rgb_image), plt.axis('off'), plt.title('RGB Image', size=10)
plt.show()
