import cv2

# Load the image in grayscale
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images
cv2.imshow("Original Image", img)
cv2.imshow("Thresholded Image", thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
