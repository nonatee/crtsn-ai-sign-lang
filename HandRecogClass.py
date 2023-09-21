import cv2 as cv
import mediapipe as mp
import time


class HandRecognnition:
    def __init__(self):
        self.cap = cv.VideoCapture(0)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        self.previous_time = 0
        self.current_time = 0
        self.fps = 0

    def fps_write(self, x, y):
        self.current_time = time.time()
        self.fps = 1 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time
        cv.putText(self.frame, "FPS: "+str(int(self.fps)), (x, y), cv.FONT_ITALIC, 0.5, (0, 0, 0), thickness=1)

    def main(self):
        is_true, self.frame = self.cap.read()
        self.imgRGB = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(self.imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, landmark in enumerate(handLms.landmark):
                    height, width, channels = self.frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    print(id, cx, cy)
                self.mpDraw.draw_landmarks(self.frame, handLms, self.mpHands.HAND_CONNECTIONS)

        self.fps_write(10,30)

        cv.imshow('SignLanguageAi', self.frame)


Main = HandRecognnition()

if __name__ == '__main__':
    while True:
        Main.main()
        if cv.waitKey(20) & 0xFF==ord('q'):
            break
