import cv2
import numpy as np

# Load image
# img = cv2.imread("data/my_test.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


marker_id = 3
marker_size_pixels = 20  # Size of the marker image in pixels
# Generate the marker image
marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_pixels)

img = cv2.imread("data/IMG_2722.jpg")

# ArUco detection
corners, ids, _ = detector.detectMarkers(img)
corners = np.intp(corners)
cv2.polylines(img, corners, True, (136, 231, 136), 1)
if ids is not None:
    marker_corners = corners[0][0]  # shape (4, 2)

    # Define the ROI relative to the marker
    # Assume OMR region starts 100 pixels right and 150 pixels down from marker
    # and is 500x300 in size in the marker's coordinate space
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
    dst_pts = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)

    # Get transformation from marker to ideal square
    M = cv2.getPerspectiveTransform(marker_pts, dst_pts)

    # Define the bounding box (relative to marker-aligned space)
    form_box = np.array(
        [[0, 200], [900, 200], [900, 1000], [0, 1000]], dtype=np.float32
    )

    # Transform back to image space
    form_box_img = cv2.perspectiveTransform(form_box[None, :, :], np.linalg.inv(M))[0]
    crop_img = img[
        np.int32(form_box_img[0][1]) : np.int32(form_box_img[2][1]),
        np.int32(form_box_img[0][0]) : np.int32(form_box_img[1][0]),
    ]
    print(form_box_img)

    # Draw the bounding box
    for i in range(4):
        pt1 = tuple(np.int32(form_box_img[i]))
        pt2 = tuple(np.int32(form_box_img[(i + 1) % 4]))
        cv2.line(img, pt1, pt2, (0, 255, 0), 5)
#
cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Window", 1200, 800)
cv2.imshow("Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
