import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(1)

# Define min and max pixel distances for scaling
min_dist = 20  # Minimum distance in pixels (fingers nearly touching)
max_dist = 300  # Maximum distance in pixels (fingers fully spread)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate Euclidean distance in pixels
            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # Scale distance to 0-180
            scaled_distance = (distance - min_dist) / (max_dist - min_dist) * 180
            scaled_distance = np.clip(scaled_distance, 0, 180)  # Clamp to 0-180

            # Display the scaled distance on the frame
            cv2.putText(frame, f"Scaled Distance: {int(scaled_distance)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw a line between thumb and index finger
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()