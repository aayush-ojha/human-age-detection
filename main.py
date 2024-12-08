import cv2 
import threading
import numpy as np


cam = cv2.VideoCapture(0)
model = cv2.dnn.readNetFromCaffe('age.prototxt', 'age.caffemodel')
AGE_GROUPS = ["(0-2)", "(3-6)", "(7-12)", "(13-19)", "(20-29)", "(30-39)", "(40-49)", "(50-59)", "(60-100)"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
shared_boxes = []

def cam_runner():
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Failed to capture image")
        
        if count % 30 == 0:
            threading.Thread(target=age_predict, args=(frame,)).start()

        count += 1

        for (x, y, w, h, age) in shared_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, age, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            print(age)

        cv2.imshow('Age Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def age_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    shared_boxes.clear()
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (227, 227))
        blob = cv2.dnn.blobFromImage(resized_face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        model.setInput(blob)
        prediction = model.forward()
        age = AGE_GROUPS[prediction[0].argmax()]
        shared_boxes.append((x, y, w, h, age))

cam_runner()


