# .caffemodel si .prototxt sunt utilizate pentru detectie
import imutils
import cv2
import numpy as np
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "sheep", "tvmonitor"]
def main() -> object:
    img = cv2.imread('imagine.jpg')
    img=imutils.resize(img,width=700)
    #Calculam inaltimea si latimea imaginii
    (h, w) = img.shape[:2]
    #Creează un blob 4-dimensional din serii de imagini
    #Opțional, redimensionează și recoltează imaginile din centru, scade valorile medii,schimbă canalele albastru și roșu
    blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)
    detector.setInput(blob)
    #personfound contine toate detectiile de persoane
    personfound = detector.forward()
    for i in np.arange(0, personfound.shape[2]):
        confidence = personfound[0, 0, i, 2]
        if confidence > 0:
            idx = int(personfound[0, 0, i, 1])
            if "person" != CLASSES[idx]:
                continue
                #Incadram persoana gasita intr-un drepunghi (rectangle)
            frame = personfound[0, 0, i, 3:10] * np.array([w, h, w, h])
            (STARTX, STARTY, ENDX, ENDY) = frame.astype("int")
            cv2.rectangle(img, (STARTX, STARTY), (ENDX, ENDY), (255, 0, 0), 3)
    cv2.imshow("Detectia pietonilor intr-o imagine", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()



