#Folosim algoritmul centroidtracker
#Acest algoritm urmareste coordonatele pietonilor din video
#Trackerul va salva coordonatele,dupa care va compara si calcula distanta dintre
#coordonatele curente si coordonatele anterioare
#Daca este sub o anumita valoare prag,va continua sa "urmareasca" persoana
#In caz contrar,nu mai continuam urmarirea si vom considera urmatoarea persoana ca fiind noua
import cv2
import datetime
import imutils
import numpy as np
from Centroidtracker import CentroidTracker
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
#Initializam trackerul
#maxDisappeared=nr. frame-uri pe care vrem ca trackerul nostru sa le astepte
#Daca persoana iese din cadru,trackerul va astepta un nr. de frame-uri egal cu maxDisappeared
#Daca se reintoarce,isi va prelua vechiul ID
tracker = CentroidTracker(maxDisappeared=3, maxDistance=80)
def NMS(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idx = np.argsort(y2)
        while len(idx) > 0:
            last = len(idx) - 1
            i = idx[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idx[:last]])
            yy1 = np.maximum(y1[i], y1[idx[:last]])
            xx2 = np.minimum(x2[i], x2[idx[:last]])
            yy2 = np.minimum(y2[i], y2[idx[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idx[:last]]
            idx = np.delete(idx, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))
def main():
    cap = cv2.VideoCapture('video2.mp4')
    start = datetime.datetime.now()
    fps = 0
    frames = 0
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=700)
        frames = frames + 1
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        personfound = detector.forward()
        #list=lista cu toate coordonatele salvate
        list = []
        for i in np.arange(0, personfound.shape[2]):
            confidence = personfound[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(personfound[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                personrectangle = personfound[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = personrectangle.astype("int")
                #In lista adaugam coordonatele persoanelor detectate de cod
                list.append(personrectangle)
                #Utilizam algoritmul NMS (Non-maximum Suppression)
        boundingboxes = np.array(list)
        boundingboxes = boundingboxes.astype(int)
        list = NMS(boundingboxes, 0.3)
        objects = tracker.update(list)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        end = datetime.datetime.now()
        time = end - start
        if time.seconds == 0:
            fps = 0.0
        else:
            fps = (frames / time.seconds)
        TEXT = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, TEXT, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("Pedestrian tracking", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
main()
