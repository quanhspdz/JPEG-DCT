import cv2
import numpy as np
import matplotlib.pyplot as plt

# dct block 8x8
def dct(array):
    result = np.zeros_like(array)
    # dct row
    for i in range(8):
        for u in range(8):
            sum=0
            cu = 1/np.sqrt(2) if u==0 else 1
            for v in range(8):
                sum += array[i][v]*np.cos((2 * v + 1) * np.pi * u / 16)
            result[i][u]=sum*cu*1/2
    #dct col
    for j in range(8):
        for u in range(8):
            sum=0
            cu = 1/np.sqrt(2) if u==0 else 1
            for v in range(8):
                sum += array[v][j]*np.cos((2 * v + 1) * np.pi * u / 16)
            result[u][j]=sum*cu*1/2
    return result

# idct block 8x8
def idct(result):
    reconstruction = np.zeros_like(result)
    #idct row
    for i in range(8):
        for v in range(8):
            sum=0
            for u in range(8):
                cu=1/np.sqrt(2) if u ==0 else 1
                sum+=result[i][u]*cu*np.cos((2*v+1)*np.pi*u/16)
            sum*=1/2
            reconstruction[i][v]=sum
    #idct col
    for j in range(8):
        for v in range(8):
            sum=0
            for u in range(8):
                cu=1/np.sqrt(2) if u ==0 else 1
                sum+=result[u][j]*cu*np.cos((2*v+1)*np.pi*u/16)
            sum*=1/2
            reconstruction[v][j]=sum
    return reconstruction

#dct function for an image
#dct function for an image
def dct_image(image):
    if len(image.shape) == 2:  # Check if the image is grayscale
        height, width = image.shape
        channels = 1
    else:  # Color image
        height, width, channels = image.shape

    block_size = 8
    blocks_w = width + (block_size - width % block_size) if width%block_size!=0 else width
    blocks_h = height + (block_size - height % block_size) if height%block_size!=0 else height

    new_image = np.zeros((blocks_h, blocks_w, channels))
    
    if channels == 1:
        new_image[:height, :width, 0] = image
    else:
        new_image[:height, :width, :] = image

    new_image = new_image.astype(float)
    new_image -= 128

    result = np.zeros_like(new_image)

    for c in range(channels):
        for i in range(0, blocks_h, block_size):
            for j in range(0, blocks_w, block_size):
                block = new_image[i:i+block_size, j:j+block_size, c]
                result[i:i+block_size, j:j+block_size, c] = dct(block)

    return result


#idct function for an image
def idct_image(result):
    height, width, channels = result.shape
    block_size = 8

    image = np.zeros((height, width, channels))

    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = result[i:i+block_size, j:j+block_size, c]
                image[i:i+block_size, j:j+block_size, c] = idct(block) + 128

    return image.clip(0, 255).astype(np.uint8)

# Read the color image
image = cv2.imread('og.jpg')

# Perform DCT on the grayscale image
result = dct_image(image)

# Perform IDCT on the result to get the reconstructed image
reconstruction = idct_image(result)

cv2.imwrite('decompressed_image.jpg', reconstruction)

# Display the original and reconstructed images
plt.gray()
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image', size=10)
#plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('DCT Image', size=10)
plt.subplot(122), plt.imshow(cv2.cvtColor(reconstruction, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Decompress Image', size=10)
plt.show()
