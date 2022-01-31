from datetime import datetime

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    start = datetime.now()

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([36, 100, 100])
    upper_green = np.array([86, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    (contours, _) = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if (area > 40):
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            frame = cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), 3)
            frame = cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow("tracking", frame)

    f = open("../output/data.txt", "a")
    stop = datetime.now()

    diff = (stop - start)
    f.write(str(diff.total_seconds() * 1000) + '\n')

    k = cv2.waitKey(5) & 0XFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
