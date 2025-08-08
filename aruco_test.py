import cv2
import numpy as np
import aruco


# marker_id = 3
# marker_size_pixels = 20  # Size of the marker image in pixels

# crop_img = img[
#     np.int32(form_box_img[0][1]) : np.int32(form_box_img[2][1]),
#     np.int32(form_box_img[0][0]) : np.int32(form_box_img[1][0]),
# ]

img = cv2.imread("data/IMG_2757.jpg")
box_height = 4.1
box_width = 4.3
aruco_side = 1
static = False
snapshot = True
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

if static:
    form_box_img = aruco.bounding_box(
        img, box_width, box_height, aruco_side, aruco.detect_aruco(img, detector)
    )

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 800)
    aruco.draw_poly(img, form_box_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cap = cv2.VideoCapture(0)  # You may need to change the device index
    if not snapshot:
        while True:
            ret, img = cap.read()

            form_box_img = aruco.bounding_box(
                img,
                box_width,
                box_height,
                aruco_side,
                aruco.detect_aruco(img, detector),
            )

            aruco.draw_poly(img, form_box_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    else:
        while True:
            ret, frame = cap.read()
            key = cv2.waitKey(1)
            frame_copy = frame.copy()
            form_box_img = aruco.bounding_box(
                frame,
                box_width,
                box_height,
                aruco_side,
                aruco.detect_aruco(frame, detector),
            )
            aruco.draw_poly(frame, form_box_img)
            if key == ord("s"):
                snapshot = frame.copy()
                form_box_img = aruco.bounding_box(
                    snapshot,
                    box_width,
                    box_height,
                    aruco_side,
                    aruco.detect_aruco(snapshot, detector),
                )
                aruco.print_snapshot(frame_copy, form_box_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
cv2.destroyAllWindows()
