#Thu thập dữ liệu
import cv2
import os

from numpy.ma.core import count

#Khai bao va bat camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4,480)
#Khai bao thư viện, file này để phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Nhap ID Khuon Mat <return> ===> ')

print("\n [INFO] Khoi tao Camera ........")
count = 0

while(True):

    ret, img = cam.read()#Bat cam
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# chuyen sang anh đen trắng
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        #Khuong vung khuon mat
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        #luu anh vao muc dataset, anh o mục đen trắng
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()

