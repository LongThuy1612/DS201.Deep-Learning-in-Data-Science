import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
from tqdm import tqdm

import os
from PIL import Image

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import ResNetForImageClassification, AutoImageProcessor

train_dir = './VinaFood21/train'
test_dir = './VinaFood21/test'

X = []
y = []
class_names = sorted(os.listdir(train_dir))
print("Các lớp:", class_names)

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)
        try:
            with open(file_path, "rb") as f:
                img = Image.open(f).convert("RGBA")  # ép tất cả về RGB (3 kênh)
                img = img.resize((224, 224))
                X.append(np.array(img))
                y.append(class_idx)
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print("Kích thước X:", X.shape)
print("Kích thước y:", y.shape)

X_test = []
y_test = []
class_names = sorted(os.listdir(test_dir))
print("Các lớp:", class_names)

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)
        try:
            with open(file_path, "rb") as f:
                img = Image.open(f).convert("RGB")  # ép tất cả về RGB (3 kênh)
                img = img.resize((224, 224))
                X_test.append(np.array(img))
                y_test.append(class_idx)
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Kích thước X:", X_test.shape)
print("Kích thước y:", y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_new = to_categorical(y_train, num_classes = 21)
y_val_new = to_categorical(y_val, num_classes = 21)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

def preprocess(X):
    tensors = []
    for img in X:
        inputs = processor(images=img, return_tensors="pt")
        tensors.append(inputs["pixel_values"])
    return torch.cat(tensors, dim=0)

X_train_t = preprocess(X_train)
X_val_t = preprocess(X_val)
X_test_t  = preprocess(X_test)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

class PretrainedResnet(nn.Module):
    def __init__(self):
        super().__init__()

        basemodel = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        self.resnet = basemodel.resnet
        self.classifier = nn.Linear(in_features=2048, out_features=21, bias=True)

    def forward(self, images: torch.Tensor):
        features = self.resnet(images).pooler_output
        features = features.squeeze(-1).squeeze(-1)
        logits = self.classifier(features)

        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PretrainedResnet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

    model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = round(accuracy_score(y_test, y_pred)*100,2)
precision = round(precision_score(y_test, y_pred, average='macro')*100,2)
recall = round(recall_score(y_test, y_pred, average='macro')*100,2)
f1 = round(f1_score(y_test, y_pred, average='macro')*100,2)

print('Accuracy test = {}%'.format(accuracy))
print('Precision test = {}%'.format(precision))
print('Recall test = {}%'.format(recall))
print('F1-macro test = {}%'.format(f1))
