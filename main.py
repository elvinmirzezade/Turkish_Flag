import cv2
import numpy as np

path = "D:\Turkey_Flag\turkey_flag.png"



# Define lower and upper bounds for red in HSV
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20000:  # Smaller threshold for lower resolution
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]

            # Optional: Check for white crescent/star in ROI here

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            flag_detected = True

    if flag_detected:
        cv2.putText(frame, "Turkish Flag Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Flag Not Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Flag Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
