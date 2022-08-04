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
    loaded_model = model.load_model('webcam_knn_model_no_pca.sav')
    # loaded_pca = pca.load_pca('pca_6.sav')

    while True:
        success, image = cap.read()
        frame = cv2.flip(image, 1)

        frame = tracker.find_hands(frame)
        tracker.draw_borders(frame)

        if tracker.results.multi_hand_landmarks:

            # cropped = tracker.slice_hand_imgs(cv2.flip(image, 1), 0)
            # plt.imshow(cropped)
            # plt.show()

            lmlist = tracker.find_positions(cv2.flip(image, 1))
            # print(lmlist)
            xlist = np.array(lmlist[:, 2])
            # print(xlist)
            ylist = np.array(lmlist[:, 3])
            # print(ylist)

            xylist = []
            for cx in xlist:
                xylist.append(cx)
            for cy in ylist:
                xylist.append(cy)

            # xylist = np.stack([xlist, ylist])

            for i in range(len(tracker.results.multi_hand_landmarks)):

                pos = np.array(xylist[i*(21*2):(i+1)*(21*2)])
                print(pos)
                pos = pos.reshape(1, -1)

                # pos_pca = loaded_pca.transform(pos)
                # pos_pca = loaded_pca.inverse_transform(pos_pca)

                predicted_letter = loaded_model.predict(pos)
                predicted_letter = letters[int(predicted_letter)]
                print(predicted_letter)

                probability = np.ravel(loaded_model.predict_proba(pos))
                # print(probability)
                probability = max(probability) * 100
                print('confidence:', probability)

                if probability >= 25:
                    tracker.display_letters(frame, i, predicted_letter, round(probability, 2))

        cv2.imshow("Signed English Translator", frame)
        cv2.waitKey(1)
