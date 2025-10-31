import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn

import os
from PIL import Image

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, Input
from keras.layers import Dense, Input, MaxPooling2D, Conv2D, BatchNormalization, \
    concatenate, GlobalAveragePooling2D, ReLU, Add
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

def residual_block(x, filters, stride=1):
    shortcut = x

    x = Conv2D(filters, (3,3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3,3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


def ResNet18(input_shape=(224,224,3), num_classes=21):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7,7), strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='ResNet18')
    return model

model_resnet18 = ResNet18(input_shape=(224,224,3), num_classes=21)
model_resnet18.compile(
    optimizer=Adam(learning_rate = 1e-4),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

history_3 = model_resnet18.fit(X_train, y_train_new, 
                               validation_data=(X_val, y_val_new),
                               batch_size=64, epochs=20)

y_pred = model_resnet18.predict(X_test)
y_pred = np.argmax(y_pred, axis = -1)

accuracy = round(accuracy_score(y_test, y_pred)*100,2)
precision = round(precision_score(y_test, y_pred, average='macro')*100,2)
recall = round(recall_score(y_test, y_pred, average='macro')*100,2)
f1 = round(f1_score(y_test, y_pred, average='macro')*100,2)

print('Accuracy test = {}%'.format(accuracy))
print('Precision test = {}%'.format(precision))
print('Recall test = {}%'.format(recall))
print('F1-macro test = {}%'.format(f1))
