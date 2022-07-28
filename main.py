import numpy as np
import pandas as pd

import cv2
import time
import mediapipe as mp

import svm
import handTracker


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    pTime = 0
    cTime = 0

    while True:
        success, image = cap.read()
        image = tracker.find_hands(image)
        tracker.draw_borders(image)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Signed English Translator", image)
        cv2.waitKey(1)
