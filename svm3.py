import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn import svm, decomposition
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

import cv2
import pickle
import os
import csv

import handTracker


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/webcam dataset'
letters = sorted(os.listdir(imgdir))


class Data:

    def __init__(self):
        self.pca = None
        self.X, self.y = [], []
        self.labels = []
        self.X_pca = []

    def load_data(self, datadir):
        index = 0

        folders = sorted(os.listdir(datadir))
        images = []
        self.labels = folders
        # print(folders)

        tracker = handTracker.HandTracker(max_hands=1)
        # file = open('collected_coordinates_ver2.csv', 'w')
        # writer = csv.writer(file)

        # separate folder for each letter
        for folder in folders:

            print("Loading images from folder", folder, "has started.")
            # imgind = -1

            for image in os.listdir(datadir + '/' + folder):

                # imgind += 1
                # # print(imgind)
                #
                # if imgind <= 1300:
                #     continue
                # elif imgind >= 1600:
                #     break

                img = cv2.imread(datadir + '/' + folder + '/' + image)
                # img = resize(img, (64, 64))
                # img = rgb2gray(img)
                # img /= 255
                #
                # self.X.append(img.flatten())
                # self.y.append(index)
                # img = imread('C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train/A/A1303.jpg')

                # plt.imshow(img)
                # plt.show()

                # print(np.shape(img))
                tracker.find_hands(img)
                # print("Found hand!")

                if tracker.results.multi_hand_landmarks:
                    lmlist = tracker.find_positions(img)
                    # print(lmlist)

                    # for i in range(len(lmlist)):
                    #     writer.writerow([image,
                    #                      'finger_id = {}'.format(lmlist[i, 1]),
                    #                      (lmlist[i, 2], lmlist[i, 3]),
                    #                      'Class = {}'.format(index)])

                    xlist = np.array(lmlist[:, 2])
                    # print(xlist)
                    ylist = np.array(lmlist[:, 3])
                    # print(ylist)

                    xylist = []
                    for cx in xlist:
                        xylist.append(cx)
                    for cy in ylist:
                        xylist.append(cy)

                    # print(xylist)
                    # print(len(xylist))

                    self.X.append(xylist)
                    self.y.append(index)
                # #     images.append('{}'.format(image))

            index += 1

        self.X = np.array(self.X)
        # print(np.shape(self.X))
        self.y = np.array(self.y)
        # print(np.shape(self.y))
        # self.y = to_categorical(self.y, len(folders))

        return self.X, self.y

    def eigenvalues(self):

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        C = np.cov(self.X, rowvar=False)

        eigenval, eigenvec = np.linalg.eig(C)
        eigens = np.insert(eigenvec.transpose(), 0, np.abs(eigenval), axis=1)
        eigens_sorted = eigens[eigens[:, 0].argsort()][::-1]
        sorted_evals = eigens_sorted[:, 0]
        print(sorted_evals)
        indices = np.arange(1, len(sorted_evals) + 1, 1, dtype=int)

        plt.plot(indices, sorted_evals, '-r', label="Eigens")
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()

        return

    def do_pca(self, d):

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.pca = decomposition.PCA(n_components=d)
        self.X_pca = self.pca.fit_transform(self.X)
        # self.X_pca = self.pca.inverse_transform(self.X_pca)

        return self.X_pca

    def save_pca(self, path):
        pickle.dump(self.pca, open(path, 'wb'))

    def load_pca(self, path):
        loaded_pca = pickle.load(open(path, 'rb'))
        self.pca = loaded_pca

        return self.pca


class SVM:

    def __init__(self):
        self.model = svm.SVC(probability=True)

    def fit(self, dataset, target):

        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                      'kernel': ['rbf', 'poly']}

        grid = GridSearchCV(self.model, param_grid)
        grid.fit(dataset, target)

        self.model = grid
        return self.model

        # svc = svm.SVC(C=100, gamma=1, kernel='rbf', probability=True)
        # fit = svc.fit(dataset, target)
        # self.model = fit
        #
        # return self.model

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load_model(self, path):
        loaded_model = pickle.load(open(path, 'rb'))
        self.model = loaded_model

        return self.model


if __name__ == '__main__':

    data = Data()
    X, y = data.load_data(imgdir)
    print("Done loading data!")

    # # run pca
    # data.eigenvalues()

    comp_num = 2
    pca = data.load_pca('pca_4_world.sav')
    X_pca = pca.transform(X)
    print(np.shape(X_pca))
    # data.save_pca('pca_{}_world.sav'.format(comp_num))
    # print("Saved PCA!")

    pickle.dump(X_pca, open('saved models/X_pca_kaggle_to_webcam.sav', 'wb'))
    pickle.dump(y, open('saved models/y_pca_kaggle_to_webcam.sav', 'wb'))
    print("Saved X_pca!")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
    # print("Split data successfully")
    #
    # model = SVM()
    # print("Fitting data to model...")
    # fitted_model = model.fit(X_train, y_train)
    # print("Done fitting!")
    # print("Best parameters: ", fitted_model.best_params_)
    #
    # model.save_model('webcam_svm_no_pca.sav')
    # print("Saved model!")
    #
    # y_pred = fitted_model.predict(X_test)
    # print("Accuracy: ", accuracy_score(y_pred, y_test)*100)
