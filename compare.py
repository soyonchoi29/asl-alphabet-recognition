import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

if __name__ == '__main__':

    X_pca_kaggle = pickle.load(open('X_pca_2_kaggle_img.sav', 'rb'))
    y_kaggle = pickle.load(open('y_kaggle_img.sav', 'rb'))

    X_pca_webcam = pickle.load(open('X_pca_2_webcam_img.sav', 'rb'))
    y_webcam = pickle.load(open('y_webcam_img.sav', 'rb'))

    imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
    letters = sorted(os.listdir(imgdir))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    markers = []
    for letter in letters:
        markers.append('${}$'.format(letter))

    # [red, blue]
    colors = ['#de3838', '#007bc3']

    # plot kaggle data (#pc = 2)
    for l, m in zip(np.unique(y_kaggle), markers):
        plt.scatter(X_pca_kaggle[y_kaggle == l, 0], X_pca_kaggle[y_kaggle == l, 1], c=colors[0], marker=m, label=l)

    # plot webcam data on same plot
    for l, m in zip(np.unique(y_webcam), markers):
        plt.scatter(X_pca_webcam[y_webcam == l, 0], X_pca_webcam[y_webcam == l, 1], c=colors[1], marker=m, label=l)

    # # plot kaggle data (#pc = 3)
    # for l, m in zip(np.unique(y_kaggle), markers):
    #     ax.scatter(X_pca_kaggle[y_kaggle == l, 0],
    #                X_pca_kaggle[y_kaggle == l, 1],
    #                X_pca_kaggle[y_kaggle == l, 2],
    #                c=colors[0], marker=m, label=l)
    #
    # # plot webcam data on same plot
    # for l, m in zip(np.unique(y_webcam), markers):
    #     ax.scatter(X_pca_webcam[y_webcam == l, 0],
    #                X_pca_webcam[y_webcam == l, 1],
    #                X_pca_webcam[y_webcam == l, 2],
    #                c=colors[1], marker=m, label=l)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    # ax.set_xlabel('PC 1')
    # ax.set_ylabel('PC 2')
    # ax.set_zlabel('PC 3')
    # plt.legend()
    plt.show()

