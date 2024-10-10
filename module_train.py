import os
import cv2 as cv
import numpy as np

people = ['christian bale','ryan gosling','ryan reynolds']
    
DIR= r'face recognition\train'
haar_cascade = cv.CascadeClassifier(r'face recognition\haar_cascade.xml')

features = []
labels =[]

def train():
    for person in people:
        path = os.path.join(DIR, person)
        lable = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            face_react = haar_cascade.detectMultiScale(gray, 1.1, 3)
            for (x,y,h,w) in face_react:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi) # type: ignore
                labels.append(lable) # type: ignore
                
train()
print('Model is trained')
# print(len(features),len(labels))

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create() # type: ignore

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)
