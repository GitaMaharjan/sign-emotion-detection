# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the trained model from the pickle file
# model_dict = pickle.load(open('./hand_model.pickle', 'rb'))
# trained_model = model_dict['hand_classifier']

# # Initialize the video capture
# cap = cv2.VideoCapture(0)

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Initialize the Hands model
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Dictionary mapping prediction index to characters
# labels_dict = {
#     0: 'ka', 1: 'kha', 2: 'ga', 3: 'gha', 4: 'nga', 5: 'ca', 6: 'cha', 7: 'ja', 8: 'jha', 9: 'nya', 
#     10: 'tta', 11: 'ttha', 12: 'dda', 13: 'ddha', 14: 'ada', 15: 'ta', 16: 'tha', 17: 'da', 18: 'dha', 
#     19: 'na', 20: 'pa', 21: 'pha', 22: 'ba', 23: 'bha', 24: 'ma', 25: 'ya', 26: 'ra', 27: 'la', 28: 'wa', 
#     29: 'sha', 30: 'ssha', 31: 'sa', 32: 'ha', 33: 'ksha', 34: 'tra', 35: 'gya', 36: 'shra'
# }

# while True:
#     # Initialize lists to store hand landmarks
#     hand_landmarks_data = []
#     x_coords = []
#     y_coords = []

#     # Read frame from camera
#     ret, frame = cap.read()

#     # Get frame dimensions
#     height, width, _ = frame.shape

#     # Convert frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame to detect hand landmarks
#     results = hands.process(frame_rgb)
    
#     # If hand landmarks are detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks on the frame
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#             # Extract x and y coordinates of hand landmarks
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_coords.append(x)
#                 y_coords.append(y)

#             # Normalize coordinates by subtracting minimum values
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 hand_landmarks_data.append(x - min(x_coords))
#                 hand_landmarks_data.append(y - min(y_coords))

#         # Pad or truncate hand landmarks data to ensure it has 84 features
#         max_features = 84
#         hand_landmarks_data_padded = hand_landmarks_data + [0] * (max_features - len(hand_landmarks_data))

#         # Get bounding box coordinates for hand region
#         x1 = int(min(x_coords) * width) - 10
#         y1 = int(min(y_coords) * height) - 10
#         x2 = int(max(x_coords) * width) - 10
#         y2 = int(max(y_coords) * height) - 10

#         # Make prediction using the trained model
#         prediction = trained_model.predict([np.asarray(hand_landmarks_data_padded)])

#         # Get the predicted character label
#         predicted_character = labels_dict[int(prediction[0])]

#         # Draw bounding box and predicted character label on the frame
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     # Display the frame
#     cv2.imshow('frame', frame)
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from the pickle file
model_dict = pickle.load(open('./hand_model.pickle', 'rb'))
trained_model = model_dict['hand_classifier']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping prediction index to characters
labels_dict = {
    0: 'ka', 1: 'kha', 2: 'ga', 3: 'gha', 4: 'nga', 5: 'ca', 6: 'cha', 7: 'ja', 8: 'jha', 9: 'nya', 
    10: 'tta', 11: 'ttha', 12: 'dda', 13: 'ddha', 14: 'ada', 15: 'ta', 16: 'tha', 17: 'da', 18: 'dha', 
    19: 'na', 20: 'pa', 21: 'pha', 22: 'ba', 23: 'bha', 24: 'ma', 25: 'ya', 26: 'ra', 27: 'la', 28: 'wa', 
    29: 'sha', 30: 'ssha', 31: 'sa', 32: 'ha', 33: 'ksha', 34: 'tra', 35: 'gya', 36: 'shra'
}

while True:
    try:
        # Initialize lists to store hand landmarks
        hand_landmarks_data = []
        x_coords = []
        y_coords = []

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        results = hands.process(frame_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract x and y coordinates of hand landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_coords.append(x)
                    y_coords.append(y)

                # Normalize coordinates by subtracting minimum values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    hand_landmarks_data.append(x - min(x_coords))
                    hand_landmarks_data.append(y - min(y_coords))

            # Pad or truncate hand landmarks data to ensure it has 84 features
            max_features = 84
            hand_landmarks_data_padded = hand_landmarks_data + [0] * (max_features - len(hand_landmarks_data))

            # Get bounding box coordinates for hand region
            x1 = int(min(x_coords) * width) - 10
            y1 = int(min(y_coords) * height) - 10
            x2 = int(max(x_coords) * width) - 10
            y2 = int(max(y_coords) * height) - 10

            # Make prediction using the trained model
            prediction = trained_model.predict([np.asarray(hand_landmarks_data_padded)])

            # Get the predicted character label
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and predicted character label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        # Display the frame
        cv2.imshow('frame', frame)
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        break

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
