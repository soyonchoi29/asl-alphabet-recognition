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

    model = svm2.SVM()
    pca = svm2.Data()
    loaded_model = model.load_model('svm_model_pca_only.sav')
    loaded_pca = pca.load_pca('pca_6.sav')

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)
        frame = tracker.find_hands(frame)
        tracker.draw_borders(frame)

        if tracker.results.multi_hand_landmarks:

            lmlist = tracker.find_positions(frame)
            # print(lmlist)
            xlist = np.array(lmlist[:, 2])
            # print(xlist)
            ylist = np.array(lmlist[:, 3])
            # print(ylist)

            # xylist = []
            # for cx in xlist:
            #     xylist.append(cx)
            # for cy in ylist:
            #     xylist.append(cy)

            xylist = np.stack([xlist, ylist])

            for i in range(len(tracker.results.multi_hand_landmarks)):

                pos = xylist[:, i*21:(i+1)*21].flatten()
                print(pos)
                pos = pos.reshape(1, -1)

                pos_pca = loaded_pca.transform(pos)
                # pos_pca = loaded_pca.inverse_transform(pos_pca)

                predicted_letter = loaded_model.predict(pos_pca)
                predicted_letter = letters[int(predicted_letter)]
                print(predicted_letter)

                probability = np.ravel(loaded_model.predict_proba(pos_pca))
                # print(probability)
                probability = max(probability) * 100
                print(probability)

                if probability >= 60:
                    tracker.display_letters(frame, i, predicted_letter, round(probability, 2))

        cv2.imshow("Signed English Translator", frame)
        cv2.waitKey(1)
