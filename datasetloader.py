import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PA100kDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        targets = torch.tensor(self.annotations.iloc[idx, 1:].values.astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets

# Define transformations for the images
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to CSV files
data = r'C:\Users\mehrs\SIH\PA-100K\data'
train_csv = 'train.csv'
val_csv = 'val.csv'
test_csv = 'test.csv'

# Create dataset instances
train_dataset = PA100kDataset(csv_file=train_csv, root_dir=data, transform=train_transforms)
val_dataset = PA100kDataset(csv_file=val_csv, root_dir=data, transform=val_transforms)
test_dataset = PA100kDataset(csv_file=test_csv, root_dir=data, transform=test_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
