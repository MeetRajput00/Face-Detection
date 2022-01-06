import cv2

image = cv2.imread("kids.jpg")
path="haarcascade_frontalface_default.xml"
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(path)

faces = face_cascade.detectMultiScale(img_gray)

print("No. of faces are "+str(len(faces)))

for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

cv2.imwrite("faces_detected.jpg", image)