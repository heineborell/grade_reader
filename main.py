import cv2
from manip import image_manip, get_center


def main():
    img_path = "data/IMG_2721.jpg"

    img = cv2.imread(img_path)
    morph = image_manip(img_path)
    result, centers = get_center(img, morph)

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1200, 800)
    cv2.imshow("Window", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
