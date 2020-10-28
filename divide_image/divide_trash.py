import numpy as np
import cv2
import os

def divideTrash(trashLocation, width):
    listLocationLabel = []

    for i in range(0, len(trashLocation)):
        centerX = (trashLocation[i][1] + trashLocation[i][3]) / 2

        locationLabel = 0
        if centerX >= 0 and centerX <= width / 3:
            locationLabel = 1
        elif centerX > width / 3 and centerX <= (width * 2) / 3:
            locationLabel = 2
        else:
            locationLabel = 3

        listLocationLabel.append(locationLabel)

    return listLocationLabel

if __name__ == '__main__':
    image = cv2.imread('box.jpg', 1)

    height, width, channel = image.shape

    trashLocation = [[ 46, 520, 142, 587],
                    [104, 123, 157, 185]]
    trashLabel = [1, 3]

    resultLocationLabel = divideTrash(trashLocation, width)

    print(resultLocationLabel)