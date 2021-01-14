#Vom folosi alte module de detectie doar pentru fata
#Se realizeaza aproximativ la fel cu detectia pietonilor intr-un video
import imutils
import numpy as np
import cv2
import datetime
protopath = "deploy.prototxt"
modelpath = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
def main():
    video = cv2.VideoCapture('video4.mp4')
    start = datetime.datetime.now()
    FPS = 0
    frames = 0
    while True:
        ret, frame = video.read()
        frame = imutils.resize(frame, width=500)
        frames = frames + 1
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        detector.setInput(blob)
        detection = detector.forward()
        for i in np.arange(0, detection.shape[2]):
            confidence = detection[0, 0, i, 2]
            if confidence > 0.5:
                faceframe = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = faceframe.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        end = datetime.datetime.now()
        time = end - start
        if time.seconds == 0:
            FPS = 0.0
        else:
            FPS = (frames / time.seconds)
        format = "FPS : {:.2f}".format(FPS)
        cv2.putText(frame, format, (5, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        cv2.imshow("Detectia fetei intr-un video", frame)
        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
main()
