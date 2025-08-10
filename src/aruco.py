import cv2
import numpy as np
from manip import image_manip, get_center, centers_to_numbers


def detect_aruco(img, detector):
    # ArUco detection
    corners, ids, _ = detector.detectMarkers(img)
    return corners, ids


def bounding_box(img, box_width, box_height, aruco_side, *args):
    # need integer corners for opencv polylines
    corners_int = np.intp(args[0][0])
    ids = args[0][1]
    if ids is not None:
        cv2.polylines(img, corners_int, True, (0, 0, 255), 2)
        if ids is not None:
            # Aruco perimeter which will used to fix the bounding box size
            aruco_perimeter = cv2.arcLength(args[0][0][0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 4 * aruco_side

            # Define the bounding box (relative to marker-aligned space)
            y_shift = 0.8 * pixel_cm_ratio

            upper_left_x = corners_int.tolist()[0][0][3][0]  # this is in pixels
            upper_left_y = corners_int.tolist()[0][0][3][1]  # this is in pixels

            form_box_img = np.array(
                [
                    [upper_left_x, upper_left_y + y_shift],
                    [upper_left_x + box_width * pixel_cm_ratio, upper_left_y + y_shift],
                    [
                        upper_left_x + box_width * pixel_cm_ratio,
                        upper_left_y + pixel_cm_ratio * box_height,
                    ],
                    [upper_left_x, upper_left_y + pixel_cm_ratio * box_height],
                ],
                dtype=np.float32,
            )

        return form_box_img


def draw_poly(img, form_box_img):
    if form_box_img is not None:
        pts = np.intp(form_box_img).reshape(-1, 1, 2)
        cv2.fillConvexPoly(img, pts, (0, 255, 0))
        cv2.imshow("Window", img)


def draw_box(img, form_box_img):
    if form_box_img is not None:
        print(np.intp(form_box_img))
        # # Draw the bounding box
        for i in range(4):
            pt1 = tuple(np.int32(form_box_img[i]))
            pt2 = tuple(np.int32(form_box_img[(i + 1) % 4]))
            cv2.line(img, pt1, pt2, (0, 0, 255), 4)
        cv2.imshow("Window", img)


def print_snapshot(img, form_box_img):
    if form_box_img is not None:
        # Convert polygon points to int and reshape for OpenCV
        pts = np.intp(form_box_img).reshape(-1, 1, 2)

        # Create empty mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Fill polygon in mask
        cv2.fillConvexPoly(mask, pts, 255)

        # Convert BGR image to BGRA (adds alpha channel)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Apply mask to alpha channel
        rgba[:, :, 3] = mask

        # Get bounding box around the non-zero mask pixels
        coords = cv2.findNonZero(mask)
        if coords is not None:  # safety check
            x, y, w, h = cv2.boundingRect(coords)
            cropped = rgba[y : y + h, x : x + w]
        else:
            cropped = None  # nothing detected

        # Optionally save
        if cropped is not None:
            cv2.imwrite("cropped.png", cropped)

        morph = image_manip(cropped)
        result, centers = get_center(cropped, morph)
        centers_to_numbers(result, centers)

        cv2.namedWindow("Snapshot", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Snapshot", 1200, 800)
        cv2.imshow("Snapshot", result)
