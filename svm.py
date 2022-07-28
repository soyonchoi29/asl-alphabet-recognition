import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn import svm, decomposition
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

import pickle
import os


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
letters = sorted(os.listdir(imgdir))


class Data:

    def __init__(self):
        self.X, self.y = [], []
        self.labels = []
        self.X_pca = []

    def load_data(self, datadir):
        index = 0

        folders = sorted(os.listdir(datadir))
        self.labels = folders
        # print(folders)

        # separate folder for each letter
        for folder in folders:

            print("Loading images from folder ", folder, " has started.")
            imgind = 0

            for image in os.listdir(datadir + '/' + folder):

                if (imgind > 50):
                    break

                img = imread(datadir + '/' + folder + '/' + image)
                img = resize(img, (64, 64))
                img = rgb2gray(img)
                img /= 255
                self.X.append(img.flatten())
                self.y.append(index)

                imgind += 1

            index += 1

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        return self.X, self.y

    def get_label(self, index):
        return self.labels[index]

    def eigenvalues(self):

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        pca = decomposition.PCA()
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

        pca = decomposition.PCA(n_components=d)
        self.X_pca = pca.fit_transform(self.X)
        self.X_pca = pca.inverse_transform(self.X_pca)

        return self.X_pca


class SVM:

    def __init__(self):
        self.model = None

    def fit(self, dataset, target):

        # param_grid = {'C': [0.1, 1, 10, 100],
        #               'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        #               'kernel': ['rbf', 'poly']}
        #
        # grid = GridSearchCV(self.model, param_grid)
        # grid.fit(dataset, target)
        #
        # self.model = grid
        # return self.model

        svc = svm.SVC(C=100, gamma=0.0001, kernel='rbf', probability=True)
        fit = svc.fit(dataset, target)
        self.model = fit

        return self.model

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load_model(self, path):
        loaded_model = pickle.load(open(path, 'rb'))
        self.model = loaded_model

        return self.model


"""
    
if __name__ == '__main__':

    data = Data()
    X, y = data.load_data(imgdir)
    print("Done loading data!")

    # run pca
    data.eigenvalues()
    X_pca = data.do_pca(30)
    print(np.shape(X_pca))

    # display images as plots
    fig, axes = plt.subplots(2, 10, figsize=(10, 6))
    ax = axes.ravel()

    for i in range(10):
        to_show = X[i].reshape(64, 64)
        ax[i].imshow(to_show, cmap='gray')
    for i in range(10):
        to_show = X_pca[i].reshape(64, 64)
        ax[i+10].imshow(to_show, cmap='gray')

    fig.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=77)
    print("Split data successfully")

    model = SVM()
    print("Fitting data to model...")
    fitted_model = model.fit(X_train, y_train)
    print("Done fitting!")
    # print("Best parameters: ", fitted_model.best_params_)

    model.save_model('svm_model.sav')

    y_pred = fitted_model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_pred, y_test)*100)
    
"""
