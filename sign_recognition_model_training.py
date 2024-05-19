import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load hand data from pickle file
hand_data_dict = pickle.load(open('./data.pickle', 'rb'))

# Determine maximum number of features
max_features_count = max(len(sample) for sample in hand_data_dict['hand_data'])

# Pad or truncate features to the maximum length
hand_features = np.array([sample + [0] * (max_features_count - len(sample)) for sample in hand_data_dict['hand_data']])

hand_labels = np.asarray(hand_data_dict['labels'])

# Split hand data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(hand_features, hand_labels, test_size=0.2, shuffle=True, stratify=hand_labels)

# Initialize and train Random Forest classifier
hand_classifier = RandomForestClassifier()
hand_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = hand_classifier.predict(x_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy
print('{}% of hand samples were classified correctly!'.format(accuracy * 100))

# Save the trained model to a pickle file
with open('hand_model.pickle', 'wb') as f:
    pickle.dump({'hand_classifier': hand_classifier}, f)
