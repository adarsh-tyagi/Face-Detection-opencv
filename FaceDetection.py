import cv2
from random import randrange

cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# image face detection

image = cv2.imread("avengers.png")

grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

points = cascade_classifier.detectMultiScale(grayscale_img)
print(points)
for (x, y, w, h) in points:
    cv2.rectangle(image, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)
cv2.imshow("FaceDetection Image", image)

cv2.waitKey()

# real time face detection using webcam

webcam = cv2.VideoCapture(0)
# in videocapture function, instead of zero use video path to detect faces in videos
while True:
    success, frame = webcam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = cascade_classifier.detectMultiScale(gray_frame)

    print(points)
    
    for (x, y, w, h) in points:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

    cv2.imshow("VideoFaceDetection", frame)
    
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()

print("Program ran successfully")
