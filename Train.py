#Tu training ra 1 model
import cv2
import numpy as np
from PIL import Image
import os


path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# set label của bức cảnh
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    # Để lấy id để mình get label của từng bức ảnh trong dataset
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Dang trainning du lieu ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n [INFO] {0} khuon mat duoc train. Thoat".format(len(np.unique(ids))))



