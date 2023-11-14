  # Plotting the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    # Plotting the decompressed image
    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_image)
    plt.title('Image after decompression')
    plt.xticks([]), plt.yticks([])

    plt.show()
