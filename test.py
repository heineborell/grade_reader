import cv2

# Load the image in grayscale
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, img = cap.read()

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 800)
    cv2.imshow("Window", img)
    key = cv2.waitKey(1)  # wait 1 ms then move to the next frame
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
