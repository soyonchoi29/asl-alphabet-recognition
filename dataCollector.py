import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import time

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

import svm2
import handTracker


if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    num_pics = 20
    letter = 'Z'
    index = 11

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)
        frame = tracker.find_hands(frame)

        cv2.imshow("Webcam (Press Space to Capture)", frame)
        k = cv2.waitKey(1)

        if k % 256 == 32:
            tracker.draw_borders(frame)

            if index < 20 and tracker.results.multi_hand_landmarks:

                cropped = tracker.slice_hand_imgs(cv2.flip(image, 1), 0)
                # plt.imshow(cropped)
                # plt.show()

                file_dir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/webcam dataset/{}'.format(letter)
                os.chdir(file_dir)

                file = '{}{}.png'.format(letter, index)
                plt.imsave(file, cropped)
                print("{} written!".format(file))

                index += 1

