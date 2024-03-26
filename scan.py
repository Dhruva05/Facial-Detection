import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    valid, photo = cap.read()

    if not valid:
        break
   
    gray_frame = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
   
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

    cv2.putText(photo, 'No Face', (0, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
    
    if len(faces) > 0:
        face_x, face_y, face_w, face_h = faces[0]
        face_roi = face_roi = photo[face_y:face_y + face_h, face_x:face_x + face_w]

        face_roi = cv2.resize(face_roi, (200, 200))
        photo[:200, :200] = face_roi
        cv2.putText(photo, 'Face Detected', (0, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('Video', photo)
    

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
   
cap.release()
cv2.destroyAllWindows()