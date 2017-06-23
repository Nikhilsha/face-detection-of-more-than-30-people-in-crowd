
import time 
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cam=cv2.VideoCapture(0);
#img=np.cam
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('test21.jpg')
#equ = cv2.equalizeHist(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('s',gray1)
#cv2.waitKey(0)
#gray = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
#cv2.imshow('s',gray)
cv2.waitKey(0)
#cv2.imshow("cropped",gray)
#cv2.imshow("",img)
faces = face_cascade.detectMultiScale(gray, 1.2,2, minSize=(0,0))#for selfie(gray, 1.2decrease if extra face detected(),1(increase for primary camera), minSize=(0,0))
print(faces)
i=0
for(x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    crop_img = img[y:y+h, x:x+w]
    #cv2.imshow("im8g",crop_img)
    #cv2.saveimage('pic'+str(i)+'.jpg', crop_img)
    cv2.imwrite("image%i.jpg"%i,crop_img )
    if cv2.waitKey(10)==len(faces):
        break
    i += 1
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

