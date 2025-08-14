import cv2
import numpy as np


def image_manip(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold on white color (preparing the img for morphology)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    # # apply morphology close this is to kill anything other than the answer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
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
    for cntr in contours:
        M = cv2.moments(cntr)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
        cv2.circle(result, (cx, cy), 2, (0, 0, 255), 3)
    return result, centers


def get_first_digit(no):
    first_digit_str = str(abs(no))[
        0
    ]  # Get the first character of the string representation
    first_digit = int(first_digit_str)  # Convert to integer
    return first_digit


def distance(ordered_list, centers):
    if centers is not None:
        centers = np.array(centers).reshape(-1, 2)

        fin_mat = np.zeros((10, 9))
        for i, center in enumerate(centers):
            distance_mat = []
            print(f"center {i}", center)
            for row in ordered_list:
                distance_row = []
                for circle in row:
                    distance_row.append(
                        (circle[0] - center[0]) ** 2 + (circle[1] - center[1]) ** 2
                    )
                    distance_row = [int(x) for x in distance_row]
                distance_mat.append(distance_row)
            distance_mat = (distance_mat == np.array(distance_mat).min()) * 1
            fin_mat = fin_mat + distance_mat

        print(fin_mat)


def centers_to_numbers(centers, circles, size):
    circle_list = [tuple(i[:2].tolist()) for i in circles[0, :]]
    circle_list = sorted(circle_list, key=lambda x: x[1])
    if len(circle_list) == size:
        circle_list = np.array(circle_list).reshape(10, 9, 2)
        ordered_list = []
        for j in circle_list:
            ordered_list.append(sorted(j.tolist(), key=lambda x: x[0]))

        (np.array(ordered_list))
        distance(ordered_list, centers)
