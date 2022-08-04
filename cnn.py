import numpy as np
import os
import cv2

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle


# decided not to utilize!

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
            img = resize(img, (64, 64))
            img /= 225

            images.append(img.flatten())
            target.append(index)

        index += 1

    images = np.array(images)
    images = images.astype('float32')
    images /= 255  # normalize data just in case?

    target = np.array(target)

    return images, target


class CNN:

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
        img = cv2.resize(img, (64, 64))
        img = np.array(img)
        img /= 255.0
        pred = self.model.predict(img)
        return classes[np.argmax(pred)], pred

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load_model(self, path):
        loaded_model = pickle.load(open(path, 'rb'))
        self.model = loaded_model

        return self.model


if __name__ == '__main__':

    X, y = load_data(imgdir)
    print("Done loading data!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train_hot = to_categorical(y_train, num_classes=30)
    y_test_hot = to_categorical(y_test, num_classes=30)

    from sklearn.utils import shuffle

    X_train, y_trainHot = shuffle(X_train, y_train, random_state=13)
    X_test, y_testHot = shuffle(X_test, y_test, random_state=13)
    X_train = X_train[:30000]
    X_test = X_test[:30000]
    y_trainHot = y_trainHot[:30000]
    y_testHot = y_testHot[:30000]

    # create the model
    model = CNN(Sequential())
    model = model.create_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train)

    model.save_model('cnn_model.sav')

    # predict using the model
    label_pred, y_pred = model.predict(letters, X_test)
    print("Done predicting!")

    print("Accuracy: ", accuracy_score(y_pred, y_test)*100)

