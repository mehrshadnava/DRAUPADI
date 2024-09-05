import torch
import torch.nn as nn
import cv2
from torchvision import transforms

# Define the model architecture (same as the trained model)
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
        
        # Adjust the input size of the Linear layer
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),  # Updated input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classifier(x)
        return x

# Load the trained model
model_path = r'C:\Users\mehrs\SIH\PA-100K\gender_recognition_model.pth'
model = GenderRecognitionModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust to expected input size for your model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict gender from a detected region (e.g., a potential human figure)
def gender_prediction(region):
    # Apply transformations
    tensor_region = transform(region).unsqueeze(0)
    
    # Predict gender
    with torch.no_grad():
        output = model(tensor_region)
        prediction = torch.round(output).item()
    return "Male" if prediction < 0.5 else "Female"

# Open the webcam
cap = cv2.VideoCapture(0)

# Set up the SimpleBlobDetector to find regions in the frame that might represent people
detector = cv2.SimpleBlobDetector_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally resize the frame for faster processing (e.g., width=640)
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert to grayscale (needed for blob detection)
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect blobs (simulating person detection)
    keypoints = detector.detect(gray_frame)

    # Loop over detected blobs (potential people)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)

        # Define region of interest (ROI) around each detected blob
        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(frame_resized.shape[1], x + size // 2), min(frame_resized.shape[0], y + size // 2)
        roi = frame_resized[y1:y2, x1:x2]

        # Predict gender for the ROI
        gender = gender_prediction(roi)

        # Draw a rectangle around the ROI (detected person)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the gender prediction
        cv2.putText(frame_resized, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gender Recognition', frame_resized)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
