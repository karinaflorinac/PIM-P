# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()

        self.maxDisappeared = maxDisappeared

        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # pentru a anula un ID obiect ștergem ID-ul obiectului din ambele dicționare
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]

    def update(self, rects):

        # verificam dacă lista dreptunghiurilor casetei de delimitare a intrării este gol
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # dacă am atins un număr maxim de cadre consecutive,
                # în care un anumit obiect a fost marcat lipsă,
                # îl marcăm
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # revenim de indata ce nu mai avem centroizi sau informații de urmărire
            return self.bbox

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # utilizam coordonatele casetei de delimitare pentru a obține centroidul
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])
        return self.bbox

