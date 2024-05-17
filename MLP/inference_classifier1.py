import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture and MediaPipe components
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define gesture labels
labels_dict = {
    0: 'Hello', 1: 'Ok', 2: 'Rejected', 3: 'I love you',
    4: 'Call me', 5: 'Peace', 6: 'Wait', 7: 'Why', 8: 'Perfect'
}

# Variables for accuracy tracking
total_frames = 0
correct_predictions = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

            # Extract normalized hand landmark coordinates
            hand_coords = []
            for landmark in hand_landmarks.landmark:
                hand_coords.append(landmark.x)
                hand_coords.append(landmark.y)

            # Ensure correct feature vector size (84 features)
            if len(hand_coords) == 42:  # 21 landmarks * 2 coordinates
                 # Make prediction using the model
                prediction = model.predict_proba([np.array(hand_coords)])
                confidence = np.max(prediction)  # Get the maximum confidence score
                predicted_index = np.argmax(prediction)  # Get the index of the highest probability

                # Fetch the predicted label from labels_dict
                predicted_label = labels_dict[predicted_index]


                # Display predicted gesture and confidence on frame
                cv2.putText(frame, f"Predicted: {predicted_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
