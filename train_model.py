import torch
import torch.nn as nn
import torch.optim as optim
from datasetloader import PA100kDataset, train_transforms, val_transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Initialize the model, loss function, and optimizer
model = GenderRecognitionModel().to(device)  # Move model to GPU
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paths to data and CSV files
data_dir = r'C:\Users\aarusavla\codes\SIH\DRAUPADI\PA-100K\data'
train_csv = r'C:\Users\aarusavla\codes\SIH\DRAUPADI\PA-100K\train.csv'
val_csv = r'C:\Users\aarusavla\codes\SIH\DRAUPADI\PA-100K\val.csv'

# Initialize datasets with correct paths
full_train_dataset = PA100kDataset(csv_file=train_csv, root_dir=data_dir, transform=train_transforms)
full_val_dataset = PA100kDataset(csv_file=val_csv, root_dir=data_dir, transform=val_transforms)

# Create a subset of 10% of the training data
indices = np.arange(len(full_train_dataset))
np.random.shuffle(indices)
subset_size = int(0.1* len(full_train_dataset))
subset_indices = indices[:subset_size]

# Create subsets
train_dataset = Subset(full_train_dataset, subset_indices)
val_dataset = full_val_dataset  # Optionally, you can also create a subset for validation

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data and labels to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels[:, 0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Move data and labels to GPU
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels[:, 0])
                val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f'Validation Loss: {epoch_val_loss:.4f}')

# Train the model
train_model(model, train_loader, val_loader)

# Save the trained model
torch.save(model.state_dict(), r'C:\Users\aarusavla\codes\SIH\DRAUPADI\PA-100K\gender_recognition_model1.pth')
