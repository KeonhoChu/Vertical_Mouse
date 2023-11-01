import streamlit as st
import cv2
import numpy as np
import hand_detector as hd
import pyautogui

# Set up your constants and variables
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
plocX, plocY = 0, 0

# Initialize your OpenCV video capture, hand detector, and screen size
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = hd.handDetector(detectionCon=0.7)
wScr, hScr = pyautogui.size()

# Streamlit application
st.title("Virtual Mouse")

# Display the processed video stream
stframe = st.empty()

# Run the application loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the frame horizontally

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    output = img.copy()

    if len(lmList) != 0:
        # Extract hand landmarks
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Detect finger status
        fingers = detector.fingersUp()

        # Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move Mouse with x-coordinate correctly adjusted
            pyautogui.moveTo(clocX, clocY)
            cv2.circle(img, (x1, y1), 6, (255, 28, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Both Index and middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 6, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # Display the processed frame using Streamlit
    stframe.image(img, channels="BGR", use_column_width=True)

# Clean up resources (this code is not executed due to Streamlit's single execution model)
cap.release()
cv2.destroyAllWindows()