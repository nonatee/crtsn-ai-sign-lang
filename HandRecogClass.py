import cv2 as cv
import mediapipe as mp
import time


class HandRecognnition:
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.cap = cv.VideoCapture(0)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils

        self.infos = {}

        self.previous_time = 0
        self.current_time = 0
        self.fps = 0

    def fps_write(self, x, y):
        self.current_time = time.time()
        self.fps = 1 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time
        cv.putText(self.frame, "FPS: "+str(int(self.fps)), (x, y), cv.FONT_ITALIC, 0.5, (0, 0, 0), thickness=1)

    def get_info(self, handlms):
        for id, landmark in enumerate(handlms.landmark):
            height, width, channels = self.frame.shape
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            self.infos = {
            "ID": id,
            "x_axis" : cx,
            "y_axis" : cy
            }
            print(self.infos)

    def main(self, draw=True, infos=True):
        is_true, self.frame = self.cap.read()
        self.imgRGB = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(self.imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                if draw & infos:
                    self.get_info(handLms)
                    self.mpDraw.draw_landmarks(self.frame, handLms, self.mpHands.HAND_CONNECTIONS)

                elif draw:
                    self.mpDraw.draw_landmarks(self.frame, handLms, self.mpHands.HAND_CONNECTIONS)

                elif infos:
                    self.get_info(handLms)

        self.fps_write(10, 30)

        cv.imshow('SignLanguageAi', self.frame)


Main = HandRecognnition()

if __name__ == '__main__':
    while True:
        Main.main()
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
