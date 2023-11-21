import cv2 as cv
import os
import time
import uuid

PATH = "C:\\Users\\cumab\\Downloads\\RealTimeObjectDetection\\Tensorflow\\workspace\\images\\collectedimages"

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_imgs = 15

for label in labels:
    os.mkdir("C:\\Users\\cumab\\Downloads\\RealTimeObjectDetection\\Tensorflow\\workspace\\images\\collectedimages" + label)
    cap = cv.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        print("oke")
        imagename = os.path.join(PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv.imwrite(imagename,frame)
        cv.imshow('frame', frame)
        time.sleep(2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
