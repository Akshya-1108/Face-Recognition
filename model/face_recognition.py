import cv2 as cv
import numpy as np

def resize(frame , scale = 0.75):
    width = int(frame.shape[0] * scale)
    height = int(frame.shape[1] * scale)
    
    dimension = (height, width)
    
    return cv.resize(frame , dimension , interpolation=cv.INTER_AREA)

people = ['christian bale','ryan gosling','ryan reynolds']

haar_cascade= cv.CascadeClassifier(r'face recognition\haar_cascade.xml')

face_recognizer= cv.face.LBPHFaceRecognizer_create() # type: ignore 
face_recognizer.read(r'face recognition\face_trained.yml')

img1 = cv.imread(r'face recognition\testing\cristian bale\images (6).jpeg')
img = resize(img1, 1.5)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_react = haar_cascade.detectMultiScale(gray, 1.8, 5)

for (x,y,w,h) in face_react:
    face_roi = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(face_roi)
    
    print(f'{people[label]} recognized with {confidence} confidence')
    
    cv.putText(img, str(people[label]), (20,20,), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+h,y+w), (150,0,255), 2)
print(people[label])
cv.imshow("image",img)

cv.waitKey(0)