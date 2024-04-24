import cv2
import numpy as np
import math
import time

def resize_and_pad(img, size):
    if img is None:
        return np.zeros((size, size, 3), np.uint8)  # Return a blank image if input is None
    
    h, w, _ = img.shape
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = size
        new_h = math.ceil(size / aspect_ratio)
    else:
        new_h = size
        new_w = math.ceil(size * aspect_ratio)

    resized_img = cv2.resize(img, (new_w, new_h))
    padded_img = np.ones((size, size, 3), np.uint8) * 255
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    padded_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_img

    return padded_img

cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
counter = 0
folder = "Data/Okay"

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hand = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(hand)

        if w > 0 and h > 0:  # Check if width and height are valid
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            if imgCrop.size != 0:  # Check if imgCrop is not empty
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
