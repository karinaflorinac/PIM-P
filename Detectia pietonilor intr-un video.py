import numpy as np
import datetime
import cv2
import imutils
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
def main():
    video = cv2.VideoCapture('video1.mp4')
    start = datetime.datetime.now()
    FPS = 0
    frames = 0
    while True:
        #Citim frame-urile din video-ul nostru
        ret, frame = video.read() #Citim toate frame-urile
        frame = imutils.resize(frame, width=1000)
        frames = frames + 1
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        personfound = detector.forward()
        for i in np.arange(0, personfound.shape[2]):
            confidence = personfound[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(personfound[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                personframe = personfound[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = personframe.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 2)
        end = datetime.datetime.now()
        time = end - start
        if time.seconds == 0:
            FPS = 0.0
        else:
            #Numarul de frame-uri/secunda
            FPS = (frames / time.seconds)
        text = "FPS: {:.2f}".format(FPS)
        cv2.putText(frame, text, (5, 30), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
        cv2.imshow("Detectia pietonilor intr-un video", frame)
        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
main()
