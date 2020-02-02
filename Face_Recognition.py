import cv2
import sys

# user Args
imagePath = sys.argv[1]
cascPath = 'haarcascade_frontalface_default.xml'

# classifier (haar Cascade)
faceCascade = cv2.CascadeClassifier(cascPath)

# Input Image

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# finding Faces in images
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces Found", image)
cv2.waitKey(0)
