import numpy as np
import cv2
import os
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

import svm2
import handTracker

if __name__ == '__main__':

    img = imread('C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train/H/H1166.jpg')
    letters = sorted(os.listdir('C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'))

    # plt.imshow(img)
    # plt.show()

    tracker = handTracker.HandTracker(max_hands=1)
    img = tracker.find_hands(img)

    plt.imshow(img)
    plt.show()

    model = svm2.SVM()
    pca = svm2.Data()
    loaded_model = model.load_model('svm_model_no_pca_world_grid_w_z_coord.sav')
    # loaded_pca = pca.load_pca('pca_6_world.sav')

    lmlist = tracker.find_positions(img)
    # print(lmlist)
    xlist = np.array(lmlist[:, 2])
    # print(xlist)
    ylist = np.array(lmlist[:, 3])
    # print(ylist)
    zlist = np.array(lmlist[:, 4])

    xyzlist = []
    for cx in xlist:
        xyzlist.append(cx)
    for cy in ylist:
        xyzlist.append(cy)
    for cz in zlist:
        xyzlist.append(cz)

    print(xyzlist)
    pos = np.array(xyzlist)
    pos = pos.reshape(1, -1)

    # pos_pca = loaded_pca.transform(pos)
    # pos_pca = loaded_pca.inverse_transform(pos_pca)

    predicted_letter = loaded_model.predict(pos)
    predicted_letter = letters[int(predicted_letter)]
    print(predicted_letter)

    probability = np.ravel(loaded_model.predict_proba(pos))
    # print(probability)
    probability = max(probability) * 100
    print('prob = ', probability)

