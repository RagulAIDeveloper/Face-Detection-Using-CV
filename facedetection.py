import cv2

#initialization the alogrithm
alg = "haarcascade_frontalface_default.xml"

# Loading the required haar-cascade xml classifier file 
haar_cascade = cv2.CascadeClassifier(alg)

# camera id initialization
cam = cv2.VideoCapture(0)

while True:
     # Reading the frame
    _,img = cam.read()
    # Converting image to grayscale 
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Applying the face detection method on the grayscale image
    face = haar_cascade.detectMultiScale(grayimg,scaleFactor=1.1,minNeighbors=3)
    
    # Iterating through rectangles of detected faces
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    
    cv2.imshow("FACEDETECTION",img)

    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
