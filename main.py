import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

import cv2
import time
import mediapipe as mp

import svm
import handTracker

if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    pTime = 0
    cTime = 0

    model = svm.SVM()
    loaded_model = model.load_model('svm_model.sav')

    while True:
        success, image = cap.read()
        image = tracker.find_hands(image)
        tracker.draw_borders(image)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime
        #
        # cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if tracker.results.multi_hand_landmarks:
            for i in range(len(tracker.results.multi_hand_landmarks)):
                img = tracker.slice_hand_imgs(image, i)
                # print(img)

                predicted_letter = loaded_model.predict(img)
                predicted_letter = letters[int(predicted_letter)]
                probability = np.ravel(loaded_model.predict_proba(img))
                probability = max(probability) * 100

                if probability >= 90:
                    plt.imshow(img.reshape(64, 64), cmap='gray')
                    plt.show()

                    tracker.display_letters(image, i, predicted_letter, probability)

        cv2.imshow("Signed English Translator", image)
        cv2.waitKey(1)
