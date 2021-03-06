import cv2
import sys

cascPath = sys.argv[1] # Takes in file as its first argument
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle around face on camera
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Break out of loop and quit
video_capture.release()
cv2.destroyAllWindows()
