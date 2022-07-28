import numpy as np
import pandas as pd

import cv2
import time
import mediapipe as mp

import svm


class HandTracker:

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, model_complexity=1, track_con=0.5):
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

    def draw_borders(self, img):

        if self.results.multi_hand_landmarks:
            for i in range(len(self.results.multi_hand_landmarks)):
                hand = self.results.multi_hand_landmarks[i]
                lmlist = []

                for finger_id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([finger_id, cx, cy])

                xlist = []
                ylist = []

                for lm in lmlist:
                    xlist.append(lm[1])
                    ylist.append(lm[2])

                # find borders of the rectangle around this hand
                min_x = min(xlist) - 20
                max_x = max(xlist) + 20
                min_y = min(ylist) - 20
                max_y = max(ylist) + 20

                rect_w = max_x - min_x
                rect_h = max_y - min_y

                cv2.rectangle(img,
                              (min_x, min_y),
                              (min_x + rect_w, min_y + rect_h),
                              (0, 0, 255),
                              2)
