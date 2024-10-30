import cv2
import mediapipe as mp
import numpy as np
# Initialize MediaPipe hand tracking module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
W = 640
H = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize the Hands module
with mp_hands.Hands(
        max_num_hands=1,  # Maximum number of hands to detect
        min_detection_confidence=0.7,  # Detection confidence threshold
        min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)

        # Convert the image to RGB for better performance
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform hand tracking
        results = hands.process(image_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw all hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Green color for landmarks
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))  # Red color for connections

                # Extract landmark coordinates
                landmarks = hand_landmarks.landmark

                # Thumb landmark index (Tip of thumb: 4th landmark)
                thumb_tip = landmarks[4]
                thumb_tip_coords = (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0]))

                # Grouped fingers: Index (8), Middle (12), Ring (16), Pinky (20)
                finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]

                # Calculate the center of mass for the four fingers
                finger_x = np.mean([f.x for f in finger_tips]) * image.shape[1]
                finger_y = np.mean([f.y for f in finger_tips]) * image.shape[0]
                finger_group_coords = (int(finger_x), int(finger_y))

                # Draw thumb tip
                cv2.circle(image, thumb_tip_coords, 10, (255, 255, 0), cv2.FILLED)  # Yellow color for thumb

                # Draw group center for other fingers
                cv2.circle(image, finger_group_coords, 10, (0, 255, 255), cv2.FILLED)  # Cyan color for grouped fingers

                # Draw line between thumb and the group of fingers (simulating a pinch gesture)
                cv2.line(image, thumb_tip_coords, finger_group_coords, (0, 0, 255), 2)  # Red line for pinch gesture

        # Display the resulting image
        cv2.imshow('Hand Tracking - Pinch Gesture', image)

        # Press 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
