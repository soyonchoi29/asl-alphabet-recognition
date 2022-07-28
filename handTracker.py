import numpy as np

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.color import rgb2gray


class HandTracker:

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, model_complexity=1, track_con=0.5):
        self.centers = None
        self.results = None
        self.mode = mode
        self.maxHands = max_hands
        self.detectionCon = detection_con
        self.modelComplex = model_complexity
        self.trackCon = track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=self.modelComplex,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_positions(self, img, hand_num=0, draw=True):

        lmlist = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for finger_id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([finger_id, cx, cy])

            if draw:
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        return lmlist

    # def draw_borders(self, img):
    #
    #     rect_centers = []
    #     image_height, image_width, _ = img.shape
    #     crops = []
    #
    #     if self.results.multi_hand_landmarks:
    #         for hand_landmarks in self.results.multi_hand_landmarks:
    #
    #             x = [landmark.x for landmark in hand_landmarks.landmark]
    #             y = [landmark.y for landmark in hand_landmarks.landmark]
    #
    #             center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
    #             rect_centers.append(center)
    #             cv2.rectangle(img,
    #                           (center[0] - 128, center[1] - 128),
    #                           (center[0] + 128, center[1] + 128),
    #                           (0, 0, 255), 2)
    #
    #             cropped = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             cropped = cropped[(center[0] - 128):(center[0] + 128), (center[1] - 128):(center[1] + 128)]
    #             if cropped.shape[0] >= 64 and cropped.shape[1] >= 64:
    #                 cropped = resize(cropped, (64, 64))
    #                 cropped = rgb2gray(cropped)
    #                 cropped /= 255
    #
    #                 cropped = cropped.flatten()
    #                 crops.append(cropped)
    #
    #     rect_centers = np.array(rect_centers)
    #     # print(np.shape(rect_centers))
    #     self.centers = rect_centers
    #
    #     crops = np.array(crops)
    #     return crops

    def draw_borders(self, img):

        rect_centers = []
        image_height, image_width, _ = img.shape

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:

                x = [landmark.x for landmark in hand_landmarks.landmark]
                y = [landmark.y for landmark in hand_landmarks.landmark]

                center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
                rect_centers.append(center)
                cv2.rectangle(img,
                              (center[0] - 90, center[1] - 90),
                              (center[0] + 90, center[1] + 90),
                              (0, 0, 255), 2)

                center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
                rect_centers.append(center)
                cv2.rectangle(img,
                              (center[0] - 90, center[1] - 90),
                              (center[0] + 90, center[1] + 90),
                              (0, 0, 255), 2)

        rect_centers = np.array(rect_centers)
        # print(np.shape(rect_centers))

        self.centers = rect_centers
        return self.centers

    def slice_hand_imgs(self, img, index):

        cropped = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = cropped[(self.centers[index, 1] - 90):(self.centers[index, 1] + 90),
                          (self.centers[index, 0] - 90):(self.centers[index, 0] + 90)]
        if np.shape(cropped)[0] >= 64 and np.shape(cropped)[1] >= 64:
            cropped = resize(cropped, (64, 64))
            cropped = rgb2gray(cropped)
            cropped /= 255

            cropped = cropped.flatten()
            cropped = cropped.reshape(1, -1)

            # plt.imshow(cropped.reshape(64, 64), cmap='gray')
            # plt.show()

            return cropped

        else:
            return None

    def display_letters(self, img, index, letter, prob):

        x, y = (self.centers[index, 0] - 128), (self.centers[index, 1] - 128)
        cv2.putText(img, '{}: {}%'.format(letter, prob), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))