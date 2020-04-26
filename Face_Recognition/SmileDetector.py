import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do detection
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, pt1 = (x, y), pt2 = (x+w, y+h), color = (255, 0, 0), thickness = 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 22 )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, pt1 = (ex, ey), pt2 = (ex+ew, ey+eh), color = (0, 255, 0), thickness = 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor = 1.7, minNeighbors = 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, pt1 = (sx, sy), pt2 = (sx+sw, sy+sh), color = (0, 0, 255), thickness = 2)
            
    return frame        

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    
video_capture = release()
cv2.destroyAllWindows()
    
    
