import cv2
import numpy as np


def detect_aruco(img, detector):
    # ArUco detection
    corners, ids, _ = detector.detectMarkers(img)
    return corners, ids


def bounding_box(img, box_width, box_height, aruco_side, *args):
    # need integer corners for opencv polylines
    corners_int = np.intp(args[0][0])
    ids = args[0][1]
    if ids is not None:
        cv2.polylines(img, corners_int, True, (136, 231, 136), 1)
        if ids is not None:
            # Aruco perimeter which will used to fix the bounding box size
            aruco_perimeter = cv2.arcLength(args[0][0][0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 4 * aruco_side

            # Destination coordinates in marker-aligned space
            dst_pts = np.array(
                [
                    [0, 0],
                    [pixel_cm_ratio, 0],
                    [1 * pixel_cm_ratio, pixel_cm_ratio],
                    [0, pixel_cm_ratio],
                ],
                dtype=np.float32,
            )

            # Get transformation from marker to ideal square
            M = cv2.getPerspectiveTransform(args[0][0][0], dst_pts)

            # Define the bounding box (relative to marker-aligned space)
            form_box = np.array(
                [
                    [0, pixel_cm_ratio],
                    [box_width * pixel_cm_ratio, pixel_cm_ratio],
                    [
                        box_width * pixel_cm_ratio,
                        pixel_cm_ratio + box_height * pixel_cm_ratio,
                    ],
                    [0, pixel_cm_ratio + box_height * pixel_cm_ratio],
                ],
                dtype=np.float32,
            )

            # Transform back to image space
            form_box_img = cv2.perspectiveTransform(
                form_box[None, :, :], np.linalg.inv(M)
            )[0]

        return form_box_img


def draw_box(img, form_box_img):
    if form_box_img is not None:
        print(form_box_img)
        # Draw the bounding box
        for i in range(4):
            pt1 = tuple(np.int32(form_box_img[i]))
            pt2 = tuple(np.int32(form_box_img[(i + 1) % 4]))
            cv2.line(img, pt1, pt2, (0, 0, 255), 5)
        #
        # cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Window", 1200, 800)
        cv2.imshow("Window", img)
