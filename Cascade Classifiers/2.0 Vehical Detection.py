import cv2

car_clsfr=cv2.CascadeClassifier('Cascades\Vehicle and pedestrain detection\cars.xml')

camera=cv2.VideoCapture('video/1.mp4')

while(True):

    ret,img=camera.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars=car_clsfr.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(100, 100))     #results=clsfr.predict(features)

    for (x,y,w,h) in cars:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,'CAR',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        
    cv2.imshow('LIVE',img)
    cv2.waitKey(1)
