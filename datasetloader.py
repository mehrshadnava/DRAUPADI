import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as v2

class PA100kDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Check if the root_dir exists
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Directory {self.root_dir} does not exist.")

        # Check if the CSV file exists
        if not os.path.isfile(csv_file):
            raise ValueError(f"CSV file {csv_file} does not exist.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        
        # Check if the image file exists
        if not os.path.isfile(img_name):
            raise FileNotFoundError(f"Image file {img_name} not found.")

        image = Image.open(img_name).convert('RGB')
        targets = torch.tensor(self.annotations.iloc[idx, 1:].values.astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets

# Define transformations for the images
train_transforms = v2.Compose([
    v2.Resize((224,224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = v2.Compose([
    v2.Resize((224,224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
