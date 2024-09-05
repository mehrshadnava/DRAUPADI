import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('gender_recognition_model.h5')

# Function to predict gender from a single frame
def gender_prediction(frame):
    # Resize and normalize the frame
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 128, 128, 3))
    
    # Predict gender
    prediction = model.predict(reshaped_frame)
    return "Male" if prediction < 0.5 else "Female"

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict gender
    gender = gender_prediction(frame)
    cv2.putText(frame, gender, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Gender Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
