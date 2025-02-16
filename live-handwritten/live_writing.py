import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import MNISTModel

# Load HSV values for pen detection
hsv_value = np.load('hsv_value.npy')
print("Loaded HSV values:", hsv_value)

# Load trained CNN model
model = MNISTModel()
model.load_state_dict(torch.load('mnist_cnn.pt', map_location=torch.device('cpu')))
model.eval()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

kernel = np.ones((5, 5), np.uint8)  # Kernel for morphological operations
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Initialize blank canvas

x1, y1 = 0, 0  # Track pen position
noise_thresh = 800  # Threshold to ignore small objects

# Preprocessing function for CNN input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),  
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Predefined HSV values for target color
    preset_lower = [85, 60, 60]   # Lower bound (H, S, V)
    preset_upper = [100, 255, 255]  # Upper bound (H, S, V)

    lower_range = np.array(preset_lower)
    upper_range = np.array(preset_upper)


    # Create mask for pen color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours of the pen tip
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2  # First point
        else:
            # Draw line from last point to new point
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 4)

        x1, y1 = x2, y2  # Update last position
    else:
        x1, y1 = 0, 0  # Reset if no pen detected

    frame = cv2.add(canvas, frame)  # Overlay drawing on camera feed

    # Extract the drawn digit
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the drawn digit
    digit_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if digit_contours:
        x, y, w, h = cv2.boundingRect(max(digit_contours, key=cv2.contourArea))
        digit_roi = thresh[y:y+h, x:x+w]  # Crop the digit

        # Resize, normalize, and convert to tensor
        digit_tensor = transform(digit_roi).unsqueeze(0)  # Shape: [1, 1, 28, 28]
        

        # Predict digit using CNN
        with torch.no_grad():
            output = model(digit_tensor)
            prediction = torch.argmax(output).item()

        # Display the predicted digit
        cv2.putText(frame, f"Predicted: {prediction}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    stacked = np.hstack((canvas, frame))  # Stack images horizontally
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))  # Scale down for display
    cv2.imshow('Mask', mask)
    digit_roi = cv2.bitwise_not(digit_roi)
    cv2.imshow("Digit Sent to Model", digit_roi)  # See processed digit



    key = cv2.waitKey(1)

    if key == 10:  # Press Enter to exit
        break

    if key == ord('c'):  # Press 'c' to clear the canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
