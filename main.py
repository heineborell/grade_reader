import cv2
from manip import image_manip

# read image make it grayscaled
img_path = "data/my_test.jpg"

img = cv2.imread(img_path)
morph = image_manip(img_path)


# # get contours
result = img.copy()
centers = []
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]
print("count:", len(contours))
print("")
for i, cntr in enumerate(contours):
    M = cv2.moments(cntr)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centers.append((cx, cy))
    cv2.circle(result, (cx, cy), 20, (136, 231, 136), -1)
    pt = (cx, cy)
    print("circle #:", i, "center:", pt)

# print list of centers
# print(centers)
#
# # save results
# cv2.imwrite("omr_sheet_thresh.png", thresh)
# cv2.imwrite("omr_sheet_morph.png", morph)
# cv2.imwrite("omr_sheet_result.png", result)
# # show results
cv2.imshow("morph", morph)
# cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
