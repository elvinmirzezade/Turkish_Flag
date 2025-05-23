import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Setup GPIO
GPIO.setmode(GPIO.BCM)

White_Led = 23
Red_Led = 24
Green_Led = 25  # New Green LED

GPIO.setup(White_Led, GPIO.OUT)
GPIO.setup(Red_Led, GPIO.OUT)
GPIO.setup(Green_Led, GPIO.OUT)

# Turn on Green LED while the program runs
GPIO.output(Green_Led, GPIO.HIGH)

# HSV red color bounds
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Camera Not Detected")
    GPIO.output(Green_Led, GPIO.LOW)
    exit()

# Reference area to estimate similarity
reference_area = 20000

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        flag_detected = False
        similarity = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 8000:
                flag_detected = True
                similarity = min((area / reference_area) * 100, 100)
                print(f"Flag Detected - Similarity: {similarity:.2f}%")
                break

        if flag_detected:
            print("Flag Detected")
            GPIO.output(White_Led, GPIO.LOW)
            GPIO.output(Red_Led, GPIO.HIGH)
            time.sleep(1)
        else:
            print("Not detected")
            GPIO.output(Red_Led, GPIO.LOW)
            time.sleep(1)
            GPIO.output(White_Led, GPIO.HIGH)

        time.sleep(0.5)

except KeyboardInterrupt:
    print("System Stopped")

finally:
    cap.release()
    GPIO.cleanup()  # This turns off all LEDs including Green
