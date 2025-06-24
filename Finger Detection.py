import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(image_rgb)

    total_fingers = 0  # Initialize total finger count

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine if it's the left or right hand
            label = handedness.classification[0].label

            # Count extended fingers for this hand
            finger_count = 0

            # Check index, middle, ring, and pinky fingers
            finger_tips = [8, 12, 16, 20]  # Landmark indices for fingertips
            finger_pips = [6, 10, 14, 18]  # Landmark indices for PIP joints
            for tip, pip in zip(finger_tips, finger_pips):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y - 0.02:
                    finger_count += 1

            # Check thumb (logic differs for left vs right hand)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            if label == 'Right':
                if thumb_tip.x < thumb_mcp.x - 0.02:
                    finger_count += 1
            else:  # Left hand
                if thumb_tip.x > thumb_mcp.x + 0.02:
                    finger_count += 1

            # Add this hand's finger count to the total
            total_fingers += finger_count

        # Display the total finger count on the frame
        cv2.putText(frame, f'Total fingers: {total_fingers}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Display a message if no hands are detected
        cv2.putText(frame, 'No hands detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with landmarks and count
    cv2.imshow('Finger Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()