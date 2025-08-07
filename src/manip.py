import cv2


def image_manip(image_path):
    # read image make it grayscaled
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # threshold on white color (preparing the img for morphology)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    # # apply morphology close this is to kill anything other than the answer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph


def get_center(og_img, morphed_img):
    # # get contours
    result = og_img.copy()
    centers = []

    # This guy return contours which are the boundaries between black and white regions. But it basically returns bunch of stuff depending on the opencv you use but using return-2 we guarantee it returns the contours
    contour_results = cv2.findContours(
        morphed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
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

    return result, centers
