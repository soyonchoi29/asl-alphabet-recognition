import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import os

# bad bc super super slow :(

imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
letters = sorted(os.listdir(imgdir))


def load_data(datadir):
    images = []
    target = []
    index = 0

    folders = sorted(os.listdir(datadir))
    # print(folders)

    # separate folder for each letter
    for folder in folders:
        for image in os.listdir(datadir + '/' + folder):

            img = imread(datadir + '/' + folder + '/' + image)
            img = resize(img, (64, 64, 3))

            images.append(img.flatten())
            target.append(index)

        index += 1

    images = np.array(images)
    target = np.array(target)

    return images, target


def svm_fit(data, target):
    svc = svm.SVC()

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
                  'kernel': ['rbf', 'poly']}

    model = GridSearchCV(svc, param_grid)
    model.fit(data, target)

    return model


if __name__ == '__main__':

    X, y = load_data(imgdir)
    print("Done loading data!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

    model = svm_fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_pred, y_test)*100)



