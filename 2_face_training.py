import cv2
import numpy as np
from PIL import Image
import os

# Path : dataset 파일 경로
path = 'C:/Users/OPENIN/Desktop/2022-s/dataset'
# LBP 알고리즘, Local-Binary-Pattern, 주변의 값을 2진수로 표현 -> 값 계산
recognizer = cv2.face.LBPHFaceRecognizer_create() # LBPH를 사용할 변수

detector = cv2.CascadeClassifier('C:/Users/openin/AppData/Local/Programs/Python/Python39/Lib/site-packages/haarcascade_frontalface_default.xml')

# images, label data 가져오기
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('C:/Users/OPENIN/Desktop/2022-s/trainer/trainer.yml')

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))