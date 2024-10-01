import pandas as pd
from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_names = os.listdir(self.data.iloc[idx, 0])
        images = [Image.open(os.path.join(self.data.iloc[idx, 0],img_name)) for img_name in img_names]
        img_array = np.stack([np.array(image) for image in images], axis=-1)
        img = Image.fromarray(img_array)

        label = int(self.data.iloc[idx,1])

        if self.transform:
            img = self.transform(img)

        return img, label



print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Number of GPUs:", torch.cuda.device_count())
print("Device Name:", torch.cuda.get_device_name(0))

list_of_models = models
resnet = models.resnet50(pretrained = True,progress = True)
num_classes = 2
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

#Train
resnet.train()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CustomDataset(csv_file="dataset_train.csv", transform=transform)
dataset_loader = DataLoader(dataset, batch_size=16, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
# Loop di training
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    resnet.train()

    for inputs, labels in dataset_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Azzerare i gradienti
        optimizer.zero_grad()

        # Forward
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataset_loader)}')

print('Training completed')
print(0)