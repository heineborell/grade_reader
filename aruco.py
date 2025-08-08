import cv2
import numpy as np


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


marker_id = 3
marker_size_pixels = 20  # Size of the marker image in pixels
# Generate the marker image
marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_pixels)

img = cv2.imread("data/IMG_2757.jpg")

# ArUco detection
corners, ids, _ = detector.detectMarkers(img)
corners = np.intp(corners)
cv2.polylines(img, corners, True, (136, 231, 136), 1)
if ids is not None:
    marker_corners = corners[0][0]  # shape (4, 2)
    # Aruco perimeter which will used to fix the bounding box size
    aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter / 4

    # Define the ROI relative to the marker
    marker_pts = np.array(
        [
            marker_corners[0],  # top-left of marker
            marker_corners[1],  # top-right
            marker_corners[2],  # bottom-right
            marker_corners[3],  # bottom-left
        ],
        dtype=np.float32,
    )

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
    M = cv2.getPerspectiveTransform(marker_pts, dst_pts)

    # Define the bounding box (relative to marker-aligned space)
    form_box = np.array(
        [
            [0, pixel_cm_ratio],
            [4.3 * pixel_cm_ratio, pixel_cm_ratio],
            [4.3 * pixel_cm_ratio, pixel_cm_ratio + 4.1 * pixel_cm_ratio],
            [0, pixel_cm_ratio + 4.1 * pixel_cm_ratio],
        ],
        dtype=np.float32,
    )

    # Transform back to image space
    form_box_img = cv2.perspectiveTransform(form_box[None, :, :], np.linalg.inv(M))[0]
    crop_img = img[
        np.int32(form_box_img[0][1]) : np.int32(form_box_img[2][1]),
        np.int32(form_box_img[0][0]) : np.int32(form_box_img[1][0]),
    ]

    # Draw the bounding box
    for i in range(4):
        pt1 = tuple(np.int32(form_box_img[i]))
        pt2 = tuple(np.int32(form_box_img[(i + 1) % 4]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 3)
#
cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Window", 1200, 800)
cv2.imshow("Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
