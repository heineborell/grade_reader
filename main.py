import cv2
from manip import image_manip

# read image make it grayscaled
img_path = "data/test_2.jpg"

img = cv2.imread(img_path)
morph = image_manip(img_path)


# # get contours
result = img.copy()
centers = []

# so this guy return contours which are the boundaries between black and white regions. But it basically returns bunch of stuff depending on the opencv you use but using return-2 we guarantee it returns the contours
contour_results = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contour_results[-2]

# here using first and zeroth moments we find the center for each contour, then draw a circle at each point
for i, cntr in enumerate(contours):
    M = cv2.moments(cntr)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centers.append((cx, cy))
    cv2.circle(result, (cx, cy), 20, (136, 231, 136), -1)
    pt = (cx, cy)
    print("circle #:", i, "center:", pt)

cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
