import cv2
import numpy as np

path = "D:\Turkey_Flag\turkey_flag.png"

# Define lower and upper bounds for red in HSV
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200000:  # Adjust threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            # Optionally, add more checks for crescent/star shapes here
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            flag_detected = True

    if flag_detected:
        cv2.putText(frame, "Turkish Flag Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Flag Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
