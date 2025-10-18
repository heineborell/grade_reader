import cv2
import aruco
import manip
import mnistpredict
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import mnist


def main():
    # this is the marker for the student id; marker_id = 3; marker_size_pixels = 20  # Size of the marker image in pixels
    # marker for grade marker_id= 4; marker_size_pixels= 20

    number_box_height = 6.4
    number_box_width = 4.3
    grade_box_height = 2
    grade_box_width = 2
    aruco_side = 1
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    static = False
    snapshot = True

    if static:
        # get the grade prediction
        grade = mnistpredict.predict_digit("cropped_grade.png")

        # get the centers circles for student numbers
        centers, circles = aruco.show_image("cropped.png")
        student_no = aruco.centers_to_numbers(centers, circles, 90)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)  # You may need to change the device index
        # Set fixed width and height
        fixed_width = 800
        fixed_height = 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, fixed_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fixed_height)
        if not snapshot:
            while True:
                _, img = cap.read()
                corners, ids, _ = detector.detectMarkers(img)
                form_box_img = aruco.bounding_box_number(
                    img, number_box_width, number_box_height, aruco_side, corners, ids
                )
                aruco.draw_poly(img, form_box_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        else:
            while True:
                _, frame = cap.read()
                key = cv2.waitKey(1)
                # get an unprocessed copy
                frame_copy = frame.copy()

                # detect one time not several
                corners, ids, _ = detector.detectMarkers(frame)
                box_num = aruco.bounding_box_number(
                    frame, number_box_width, number_box_height, aruco_side, corners, ids
                )
                box_grade = aruco.bounding_box_grade(
                    frame, grade_box_width, grade_box_height, aruco_side, corners, ids
                )

                # draw both of the polygons
                aruco.draw_poly(frame, box_num)
                aruco.draw_poly(frame, box_grade)
                cv2.imshow("Window", frame)

                if key == ord("s"):
                    snapshot = frame_copy
                    corners, ids, _ = detector.detectMarkers(snapshot)
                    form_box_img_number = aruco.bounding_box_number(
                        snapshot,
                        number_box_width,
                        number_box_height,
                        aruco_side,
                        corners,
                        ids,
                    )
                    form_box_img_grade = aruco.bounding_box_grade(
                        snapshot,
                        grade_box_width,
                        grade_box_height,
                        aruco_side,
                        corners,
                        ids,
                    )
                    aruco.crop_snapshot(frame_copy, form_box_img_number)
                    aruco.crop_snapshot(frame_copy, form_box_img_grade, number=False)
                    centers, circles = aruco.show_image("cropped_number.png")
                    student_no = aruco.centers_to_numbers(centers, circles, 90)
                    if student_no is not None:
                        grade = mnistpredict.predict_digit("cropped_grade.png")

                    print(student_no, grade)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
