import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import time

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

import svm
import handTracker


if __name__ == '__main__':

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    cap = cv2.VideoCapture(0)
    tracker = handTracker.HandTracker()

    # pTime = 0
    # cTime = 0

    model = svm.SVM()
    loaded_model = model.load_model('svm_model.sav')

    # while True:
    #     success, image = cap.read()
    #     image = cv2.flip(image, 1)
    #     image = tracker.find_hands(image)
    #
    #     # cTime = time.time()
    #     # fps = 1 / (cTime - pTime)
    #     # pTime = cTime
    #     #
    #     # cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    #     if tracker.results.multi_hand_landmarks:
    #
    #         crops = tracker.draw_borders(image)
    #         if len(crops) > 0:
    #             plt.imshow(resize(crops[0], (64, 64)), cmap='gray')
    #             plt.show
    #             predicted_indices = loaded_model.predict(crops)
    #         # print(predicted_indices)
    #
    #             for i in range(len(predicted_indices)):
    #
    #                 try:
    #                     print("trying")
    #                     predicted_letter = letters[int(predicted_indices(i))]
    #                     cv2.putText(image, predicted_letter,
    #                     (tracker.centers[i, 0]-128, tracker.centers[i, 1]-128),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #
    #                 except:
    #                     pass
    #
    #     cv2.imshow("Signed English Translator", image)
    #     cv2.waitKey(1)

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)
        frame = tracker.find_hands(frame)
        tracker.draw_borders(frame)

        if tracker.results.multi_hand_landmarks:
            for i in range(len(tracker.results.multi_hand_landmarks)):
                img = tracker.slice_hand_imgs(cv2.flip(image, 1), i)
                # print(img)

                if img.any() >= 1:

                    predicted_letter = loaded_model.predict(img)
                    predicted_letter = letters[int(predicted_letter)]
                    probability = np.ravel(loaded_model.predict_proba(img))
                    probability = max(probability) * 100

                    if probability >= 60:
                        # plt.imshow(img.reshape(64, 64), cmap='gray')
                        # plt.show()

                        tracker.display_letters(frame, i, predicted_letter, round(probability, 2))

        cv2.imshow("Signed English Translator", frame)
        cv2.waitKey(1)
