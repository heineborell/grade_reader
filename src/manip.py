import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


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


def distance(ordered_list, centers):
    if centers is not None:
        centers = np.array(centers).reshape(-1, 2)

        fin_mat = np.zeros((10, 9))
        for center in centers:
            distance_mat = []
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

        fin_mat = fin_mat.astype(int)
        if (np.sum(fin_mat, 0).sum()) == 9:
            return fin_mat
        else:
            print("The number placement of dots not correct!")


def get_digits(fin_mat):
    if fin_mat is not None:
        student_no = ""
        for j in range(0, 9):
            for i, number in enumerate(fin_mat):
                if number[j] == 1:
                    student_no = student_no + str(i)
        return student_no


def centers_to_numbers(centers, circles, size):
    circle_list = [tuple(i[:2].tolist()) for i in circles[0, :]]
    circle_list = sorted(circle_list, key=lambda x: x[1])
    if len(circle_list) == size:
        circle_list = np.array(circle_list).reshape(10, 9, 2)
        ordered_list = []
        for j in circle_list:
            ordered_list.append(sorted(j.tolist(), key=lambda x: x[0]))

        (np.array(ordered_list))
        omr_mat = distance(ordered_list, centers)
        student_no = get_digits(omr_mat)
        return student_no
    else:
        print("Number of dots is not correct!")


# 8. Functions to extract and preprocess red pencil digits
def get_grade(img_path, display=False):
    """Extract red pencil digits from image."""
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red hue range #1 (0-10)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    # Red hue range #2 (170-180)
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    # Combine both masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assume `contours` contains all detected digit contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    # Sort bounding boxes by x (left to right)
    bounding_boxes_sorted = sorted(bounding_boxes, key=lambda b: b[0])  # b[0] is x
    # Extract digit images in left-to-right order
    digits = []
    for x, y, w, h in bounding_boxes_sorted:
        digit_img = red_mask[y : y + h, x : x + w]
        digits.append(digit_img)
    if display:
        for i, digit in enumerate(digits):
            cv2.imshow(f"Digit {i}", digit)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Digit {i}")
    cv2.destroyAllWindows()
    return digits


def preprocess_char(roi, size=28):
    """Preprocess a single digit image for MNIST model.

    Converts black digit on white background -> white digit on black background (like MNIST)
    """
    # roi from get_grade is white digit on black background (from red mask)
    # We need to keep it that way (white on black) to match MNIST
    # So we DON'T invert here

    h, w = roi.shape
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    # Pad with black (0) to match MNIST's black background
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    normalized = padded.astype("float32") / 255.0
    return normalized.reshape(1, size, size, 1)


def preprocess_for_pytorch(digit_img):
    """
    Convert preprocessed digit to PyTorch tensor.

    Args:
        digit_img: Output from preprocess_char (shape: 1, 28, 28, 1)

    Returns:
        PyTorch tensor ready for model prediction
    """
    # Remove extra dimensions and get 28x28 array
    img_array = digit_img.squeeze()  # Shape: (28, 28)

    # Apply MNIST normalization
    normalized = (img_array - 0.1307) / 0.3081

    # Convert to PyTorch tensor with correct shape (1, 1, 28, 28)
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)

    return tensor, img_array


def show_processed(roi_digits):
    for i, roi in enumerate(roi_digits):
        preprocessed = manip.preprocess_char(roi)  # resize, pad, normalize
        plt.subplot(2, len(roi_digits), i + 1)
        plt.imshow(roi, cmap="gray")
        plt.title(f"Raw {i}")
        plt.axis("off")

        plt.subplot(2, len(roi_digits), i + 1 + len(roi_digits))
        plt.imshow(preprocessed[0, :, :, 0], cmap="gray")  # remove batch & channel dims
        plt.title(f"Processed {i}")
        plt.axis("off")

    plt.show()
