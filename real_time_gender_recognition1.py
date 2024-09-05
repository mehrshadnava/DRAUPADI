import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from collections import defaultdict

# Check if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the gender model architecture
class GenderRecognitionModel(nn.Module):
    def __init__(self):
        super(GenderRecognitionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load YOLOv5 model
model_path = r'C:\Users\aarusavla\codes\SIH\DRAUPADI\yolov5-master\yolov5s.pt'
model = torch.hub.load('C:/Users/aarusavla/codes/SIH/DRAUPADI/yolov5-master', 'custom', path=model_path, source='local')
model = model.to(device)  # Move the YOLO model to the selected device

# Load gender classification model
gender_model_path = r"C:\Users\aarusavla\codes\SIH\DRAUPADI\PA-100K\gender_recognition_model1.pth"
gender_model = GenderRecognitionModel().to(device)  # Move the gender model to the selected device
gender_model.load_state_dict(torch.load(gender_model_path, map_location=device))  # Load model on the selected device
gender_model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function to process the detected faces
def classify_gender(face_img):
    face_img = transform(face_img)  # Apply transformations
    face_img = face_img.unsqueeze(0).to(device)  # Add batch dimension and move to the selected device

    with torch.no_grad():
        outputs = gender_model(face_img)
        prediction = outputs.item()
        confidence = abs(prediction - 0.5) * 2  # Calculate confidence score
        gender = 'Male' if prediction < 0.5 else 'Female'
        
        # Debugging: print prediction and confidence
        print(f'Prediction: {prediction:.4f}, Gender: {gender}, Confidence: {confidence:.4f}')
        
        return gender, confidence

# Initialize face history and counters
face_history = {}
male_count = 0
female_count = 0
buffer_size = 10
buffer = {"Male": [], "Female": []}

# Load video stream
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv5
    results = model(frame)
    results.render()  # Render bounding boxes on the image

    # Extract boxes, scores, and class IDs
    boxes = []
    scores = []
    for det in results.xyxy[0]:
        # Unpack detection results
        x1, y1, x2, y2, score, class_id = map(float, det[:6])
        if class_id == 0:  # Assuming class 0 is 'person'
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(score)

    # Debugging: Print number of detected persons
    print(f"Detected persons: {len(boxes)}")

    # Reset frame-level counters
    frame_male_count = 0
    frame_female_count = 0

    # Loop through detections
    for box in boxes:
        x1, y1, x2, y2 = box
        face_img = frame[y1:y2, x1:x2]

        # Ensure the detected face is not too small or empty
        if face_img.size > 0 and face_img.shape[0] > 10 and face_img.shape[1] > 10:
            # Debugging: Print the dimensions of the face
            print(f"Face detected with size: {face_img.shape}")

            # Create a unique identifier for the face based on its bounding box
            face_id = (x1, y1, x2, y2)

            # Classify gender
            gender, confidence = classify_gender(face_img)

            # Debugging: show detected face and its classification
            cv2.imshow("Detected Face", face_img)
            print(f"Face {face_id}: {gender} with confidence {confidence:.4f}")

            # Update face history
            if face_id in face_history:
                prev_gender = face_history[face_id]
                if prev_gender != gender:
                    # Only update count if gender changes
                    if gender == 'Male':
                        frame_male_count += 1
                        frame_female_count -= 1
                    else:
                        frame_female_count += 1
                        frame_male_count -= 1
            else:
                # Add new face to history
                face_history[face_id] = gender
                if gender == 'Male':
                    frame_male_count += 1
                else:
                    frame_female_count += 1

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{gender} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Add counts to buffer and maintain buffer size
    buffer["Male"].append(frame_male_count)
    buffer["Female"].append(frame_female_count)

    if len(buffer["Male"]) > buffer_size:
        buffer["Male"].pop(0)
    if len(buffer["Female"]) > buffer_size:
        buffer["Female"].pop(0)

    # Calculate average counts from buffer
    avg_men = sum(buffer["Male"]) / len(buffer["Male"]) if buffer["Male"] else 0
    avg_women = sum(buffer["Female"]) / len(buffer["Female"]) if buffer["Female"] else 0

    # Update overall gender counts
    male_count = int(avg_men)
    female_count = int(avg_women)

    # Display counts
    cv2.putText(frame, f'Males: {male_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Females: {female_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show results
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
