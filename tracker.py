import numpy as np
import cv2

cap = cv2.VideoCapture('FP-180122-124655_14_2-180307-124002_Cam1.avi')


def mouse_clicking(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cap.release()
        cv2.destroyAllWindows()


cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_clicking)

lower_red = np.array([80, 0, 0])
upper_red = np.array([255, 50, 50])

last_box = None

while cap.isOpened():
    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb, lower_red, upper_red)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = 0
    maxsize = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > maxsize:
            maxsize = cv2.contourArea(cnt)
            best = count

        count = count + 1

    if best < len(contours):
        x, y, w, h = cv2.boundingRect(contours[best])
        last_box = [x, y, w, h]
    else:
        x, y, w, h = last_box

    cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame', rgb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
