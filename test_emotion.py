import cv2
import numpy as np
from keras.models import load_model

# Load the trained emotion detection model
emotion_model = load_model('emotion_model.h5')

# Open the default camera (0)
video_capture = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary mapping emotion indices to their corresponding labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Main loop for capturing video frames and performing emotion detection
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

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
        prediction = emotion_model.predict(reshaped_face)

        # Get the index of the predicted emotion
        emotion_index = np.argmax(prediction, axis=1)[0]

        # Get the corresponding emotion label from the dictionary
        emotion_label = emotion_labels[emotion_index]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # Draw a filled rectangle as background for the emotion label text
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)

        # Display the emotion label text above the face rectangle
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame with overlaid emotion detection results
    cv2.imshow("Emotion Detection", frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture object
video_capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
