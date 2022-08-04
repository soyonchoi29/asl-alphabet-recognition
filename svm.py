import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn import svm, decomposition
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

import pickle
import os


imgdir = 'C:/Users/soyon/Documents/Codes/ASL-Translator/dataset/train'
letters = sorted(os.listdir(imgdir))


class Data:

    def __init__(self):
        self.X, self.y = [], []
        self.labels = []
        self.pca = None
        self.X_pca = []

    def load_data(self, datadir):

        folders = sorted(os.listdir(datadir))
        self.labels = folders
        # print(folders)

        batch_size = 64
        image_size = 64
        target_dims = (image_size, image_size, 3)
        num_classes = 29

        train_len = 87000

        dataset = np.empty((train_len, image_size, image_size, 3), dtype=np.float32)
        target = np.empty((train_len,), dtype=np.int)
        cnt = 0
        for folder in folders:

            print("Loading images from folder", folder, "has started.")

            if not folder.startswith('.'):
                if folder in ['A']:
                    label = 0
                elif folder in ['B']:
                    label = 1
                elif folder in ['C']:
                    label = 2
                elif folder in ['D']:
                    label = 3
                elif folder in ['E']:
                    label = 4
                elif folder in ['F']:
                    label = 5
                elif folder in ['G']:
                    label = 6
                elif folder in ['H']:
                    label = 7
                elif folder in ['I']:
                    label = 8
                elif folder in ['J']:
                    label = 9
                elif folder in ['K']:
                    label = 10
                elif folder in ['L']:
                    label = 11
                elif folder in ['M']:
                    label = 12
                elif folder in ['N']:
                    label = 13
                elif folder in ['O']:
                    label = 14
                elif folder in ['P']:
                    label = 15
                elif folder in ['Q']:
                    label = 16
                elif folder in ['R']:
                    label = 17
                elif folder in ['S']:
                    label = 18
                elif folder in ['T']:
                    label = 19
                elif folder in ['U']:
                    label = 20
                elif folder in ['V']:
                    label = 21
                elif folder in ['W']:
                    label = 22
                elif folder in ['X']:
                    label = 23
                elif folder in ['Y']:
                    label = 24
                elif folder in ['Z']:
                    label = 25
                elif folder in ['del']:
                    label = 26
                elif folder in ['nothing']:
                    label = 27
                elif folder in ['space']:
                    label = 28
                else:
                    label = 29

                index = -1
                for image in os.listdir(datadir + '/' + folder):

                    if index > 200:
                        break
                    else:
                        index += 1

                    img_file = cv2.imread(datadir + '/' + folder + '/' + image)
                    if img_file is not None:
                        img_file = skimage.transform.resize(img_file, (image_size, image_size, 3))
                        img_arr = np.asarray(img_file).reshape((-1, image_size, image_size, 3))

                        dataset[cnt] = img_arr
                        target[cnt] = label
                        cnt += 1

            self.X = dataset
            self.y = target
            return self.X, self.y

    def get_label(self, index):
        return self.labels[index]

    def eigenvalues(self):

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        C = np.cov(self.X, rowvar=False)

        eigenval, eigenvec = np.linalg.eig(C)
        eigens = np.insert(eigenvec.transpose(), 0, np.abs(eigenval), axis=1)
        eigens_sorted = eigens[eigens[:, 0].argsort()][::-1]
        sorted_evals = eigens_sorted[:, 0]
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
        self.X_pca = self.pca.inverse_transform(self.X_pca)

        return self.X_pca

    def plot_component(self, comp):
        if comp <= len(self.pca.components_):
            mat_data = np.asmatrix(self.pca.components_[comp]).reshape(64, 64)  # reshape images
            plt.imshow(mat_data, cmap='gray')  # plot the data
            plt.xticks([])  # removes numbered labels on x-axis
            plt.yticks([])  # removes numbered labels on y-axis
            plt.title('Component {}'.format(comp))


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


if __name__ == '__main__':

    data = Data()
    data, y = data.load_data(imgdir)
    print("Done loading data!")

    # run pca
    data.eigenvalues()
    X_pca = data.do_pca(30)
    print(np.shape(X_pca))

    for i in range(30):
        data.plot_component(i)
        plt.show()

    # display images as plots
    fig, axes = plt.subplots(2, 10, figsize=(10, 6))
    ax = axes.ravel()

    for i in range(10):
        to_show = data[i].reshape(64, 64)
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
