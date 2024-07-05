import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

directory = 'Datasets/Data'
imgSize = 224
offset = 20
print(os.getcwd())

# Ensure all directories exist
if not os.path.exists(directory):
    os.mkdir(directory)

if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')

for i in range(65, 91):
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    _, img = cap.read()
    hands, img = detector.findHands(img)
    
    # Count the number of images in each directory
    count = {chr(i): len(os.listdir(os.path.join(directory, chr(i)))) for i in range(65, 91)}
    count['blank'] = len(os.listdir(os.path.join(directory, 'blank')))
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Handle cropping and resizing
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        except Exception as e:
            print("hold your hand correctly:", e)

    cv2.imshow("Image", img)
    interrupt = cv2.waitKey(10)
    
    # Save images based on key press
    for i in range(65, 91):
        letter = chr(i).lower()
        if interrupt & 0xFF == ord(letter):
            cv2.imwrite(os.path.join(directory, chr(i), str(count[chr(i)]) + '.jpg'), imgWhite)
    
    # Save blank images
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(directory, 'blank', str(count['blank']) + '.jpg'), imgWhite)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
