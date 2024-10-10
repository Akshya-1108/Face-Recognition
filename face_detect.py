import cv2 as cv
import numpy as np

img = cv.imread(r'face recognition\group of people.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('group of people', gray)

haar_cascade = cv.CascadeClassifier(r'face recognition\haar_cascade.xml')
face_detect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(f"number of faces found in the image is = {len(face_detect)}")

for (x,y,h,w) in face_detect:
    cv.rectangle(img, (x,y), (x+h,y+w), (0,255,0), thickness=1)

cv.imshow("face detected", img)

cv.waitKey(0)
cv.destroyAllWindows()