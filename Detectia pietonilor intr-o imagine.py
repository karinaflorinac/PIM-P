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
    #Calculam inaltimea si latimea
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)
    detector.setInput(blob)
    #person_detection contine toate detectiile din modelfile
    personfound = detector.forward()
    for i in np.arange(0, personfound.shape[2]):
        confidence = personfound[0, 0, i, 2]
        if confidence > 0:
            idx = int(personfound[0, 0, i, 1])
            if "person" != CLASSES[idx]:
                continue
                #Incadram persoana gasita intr-un drepunghi (rectangle)
            frame = personfound[0, 0, i, 3:10] * np.array([w, h, w, h])
            #Coordonatele persoanei gasite in imagine
            (STARTX, STARTY, ENDX, ENDY) = frame.astype("int")
            cv2.rectangle(img, (STARTX, STARTY), (ENDX, ENDY), (255, 255, 255), 3)
    cv2.imshow("Detectia pietonilor intr-o imagine", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()



