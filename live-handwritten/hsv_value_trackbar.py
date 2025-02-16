import cv2
import numpy as np

def nothing(x):
    pass

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Create a window with trackbars
cv2.namedWindow("Trackbars")

# Predefined HSV values for target color
preset_lower = [85, 60, 60]   # Lower bound (H, S, V)
preset_upper = [100, 255, 255]  # Upper bound (H, S, V)

# Create trackbars and set initial values
cv2.createTrackbar("L - H", "Trackbars", preset_lower[0], 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", preset_lower[1], 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", preset_lower[2], 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", preset_upper[0], 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", preset_upper[1], 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", preset_upper[2], 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get values from the trackbars
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # Create mask
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Display results
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask_3, frame, res))
    cv2.imshow("Trackbars", cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == 10:  # Press Enter to save values and exit
        hsv_values = [lower_range.tolist(), upper_range.tolist()]
        np.save('hsv_value.npy', hsv_values)
        print("HSV values saved:", hsv_values)
        break

cap.release()
cv2.destroyAllWindows()
