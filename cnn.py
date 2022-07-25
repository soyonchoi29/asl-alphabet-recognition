import numpy as np
import os

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#decided not to utilize!

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

        print("Loading images from folder ", folder, " has started.")
        for image in os.listdir(datadir + '/' + folder):

            img = imread(datadir + '/' + folder + '/' + image)
            img = resize(img, (64, 64, 3))

            images.append(img.flatten())
            target.append(index)

        index += 1

    images = np.array(images)
    images = images.astype('float32')
    images /= 255  # normalize data just in case?

    target = np.array(target)
    print("Shape before one-hot encoding: ", target.shape)
    target_one_hot = to_categorical(target, len(folders))
    print("Shape after one-hot encoding: ", target_one_hot.shape)

    return images, target, target_one_hot


class CNN:

    model = None

    def __init__(self, model_type):
        self.model = model_type

    def create_model(self):

        self.model.add(Conv2D(128, (3, 3), input_shape=(64, 64, 3), activation='relu'))

        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())

        self.model.add(Dropout(0.5))

        self.model.add(Dense(1024, activation='relu'))

        self.model.add(Dense(29, activation='softmax'))

        return self.model

    def predict(self, classes, img):
        class_pred = self.model.predict(img)
        return classes[np.argmax(class_pred)], class_pred


if __name__ == '__main__':

    X, y_labels, y = load_data(imgdir)
    print("Done loading data!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

    # create the model
    model = CNN(Sequential())
    model = CNN.create_model(model)

    # train the model
    model.fit()

    # predict using the model
    label_pred, y_pred = model.predict(y_labels, Sequential, X_test)
    print("Done predicting!")

    print("Accuracy: ", accuracy_score(y_pred, y_test)*100)

