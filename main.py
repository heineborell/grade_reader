import cv2
import numpy as np
import cv2.aruco as aruco


def main():
    print("Hello from grade-reader!")


if __name__ == "__main__":
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in range(4):
        img = cv2.aruco.generateImageMarker(aruco_dict, i, 100)
        cv2.imwrite(f"marker_{i}.png", img)
