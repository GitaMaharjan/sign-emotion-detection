import cv2
import numpy as np
import pickle
import mediapipe as mp
from keras.models import load_model

# Load the trained sign detection model
sign_model_dict = pickle.load(open('./hand_model.pickle', 'rb'))
sign_model = sign_model_dict['hand_classifier']

# Load the trained emotion detection model
emotion_model = load_model('emotion_model.h5')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping prediction index to characters for sign detection
sign_labels_dict = {
    0: 'ka', 1: 'kha', 2: 'ga', 3: 'gha', 4: 'nga', 5: 'ca', 6: 'cha', 7: 'ja', 8: 'jha', 9: 'nya', 
    10: 'tta', 11: 'ttha', 12: 'dda', 13: 'ddha', 14: 'ada', 15: 'ta', 16: 'tha', 17: 'da', 18: 'dha', 
    19: 'na', 20: 'pa', 21: 'pha', 22: 'ba', 23: 'bha', 24: 'ma', 25: 'ya', 26: 'ra', 27: 'la', 28: 'wa', 
    29: 'sha', 30: 'ssha', 31: 'sa', 32: 'ha', 33: 'ksha', 34: 'tra', 35: 'gya', 36: 'shra'
}

# Dictionary mapping emotion indices to their corresponding labels
emotion_labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Main loop for capturing video frames and performing both sign and emotion detection
while True:
    # Initialize lists to store hand landmarks
    hand_landmarks_data = []
    x_coords = []
    y_coords = []

    # Read frame from camera
    ret, frame = video_capture.read()

    # Get frame dimensions
    height, width, _ = frame.shape

    # Convert frame to RGB for hand detection
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

        # Make prediction using the trained sign detection model
        sign_prediction = sign_model.predict([np.asarray(hand_landmarks_data_padded)])

        # Get the predicted sign character label
        predicted_sign = sign_labels_dict[int(sign_prediction[0])]

        # Draw bounding box and predicted sign character label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    detected_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

    # Iterate through each detected face
    for (x, y, w, h) in detected_faces:
        # Extract the face region from the grayscale frame
        face_region = gray_frame[y:y+h, x:x+w]

        # Resize the face region to match the input size of the model
        resized_face = cv2.resize(face_region, (48, 48))

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the normalized face image to match the input shape of the model
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

        # Perform emotion prediction using the loaded model
        emotion_prediction = emotion_model.predict(reshaped_face)

        # Get the index of the predicted emotion
        emotion_index = np.argmax(emotion_prediction, axis=1)[0]

        # Get the corresponding emotion label from the dictionary
        emotion_label = emotion_labels_dict[emotion_index]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # Draw a filled rectangle as background for the emotion label text
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        # Display the emotion label text above the face rectangle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
