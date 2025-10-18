import cv2
import aruco
import mnistpredict


def getGradeStudentId(static):
    number_box_height = 6.4
    number_box_width = 4.3
    grade_box_height = 2
    grade_box_width = 2
    aruco_side = 1

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    student_no, grade = None, None

    if static:
        # get the grade prediction
        grade = mnistpredict.predict_digit("cropped_grade.png")

        # get the centers circles for student numbers
        centers, circles = aruco.show_image("cropped.png")
        student_no = aruco.centers_to_numbers(centers, circles, 90)

        yield student_no, grade

    else:
        cap = cv2.VideoCapture(0)
        fixed_width, fixed_height = 800, 720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, fixed_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fixed_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_copy = frame.copy()
            corners, ids, _ = detector.detectMarkers(frame)
            box_num = aruco.bounding_box_number(
                frame, number_box_width, number_box_height, aruco_side, corners, ids
            )
            box_grade = aruco.bounding_box_grade(
                frame, grade_box_width, grade_box_height, aruco_side, corners, ids
            )
            aruco.draw_poly(frame, box_num)
            aruco.draw_poly(frame, box_grade)
            cv2.imshow("Window", frame)

            key = cv2.waitKey(1) & 0xFF
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

                aruco.crop_snapshot(snapshot, form_box_img_number)
                aruco.crop_snapshot(snapshot, form_box_img_grade, number=False)

                centers, circles = aruco.show_image("cropped_number.png")
                student_no = aruco.centers_to_numbers(centers, circles, 90)
                if student_no is not None:
                    grade = mnistpredict.predict_digit("cropped_grade.png")
                    print(student_no, grade)

                yield (
                    student_no,
                    grade,
                )  # this makes it a generator looping in the background until you press q

            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
