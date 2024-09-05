import scipy.io
import csv
import os

# Path to the annotation.mat
path = r'C:\Users\mehrs\SIH\PA-100K\annotation.mat'

# Load the .mat file
mat_data = scipy.io.loadmat(path)

# Extract the required data
attributes = [attr[0][0] for attr in mat_data['attributes']]
train_images = [img[0][0] for img in mat_data['train_images_name']]
train_labels = mat_data['train_label']
val_images = [img[0][0] for img in mat_data['val_images_name']]
val_labels = mat_data['val_label']
test_images = [img[0][0] for img in mat_data['test_images_name']]
test_labels = mat_data['test_label']

# Save the train data to a CSV file
with open('train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image'] + attributes)  # Write the header row
    for image, labels in zip(train_images, train_labels):
        row = [image] + list(labels)
        writer.writerow(row)

# Save the validation data to a CSV file
with open('val.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image'] + attributes)  # Write the header row
    for image, labels in zip(val_images, val_labels):
        row = [image] + list(labels)
        writer.writerow(row)

# Save the test data to a CSV file
with open('test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image'] + attributes)  # Write the header row
    for image, labels in zip(test_images, test_labels):
        row = [image] + list(labels)
        writer.writerow(row)
