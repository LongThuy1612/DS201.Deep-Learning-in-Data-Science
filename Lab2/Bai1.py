import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn

import os
from PIL import Image

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Input, MaxPooling2D, Conv2D, \
    Flatten, Dropout, AveragePooling2D, Rescaling, GlobalAveragePooling2D
from keras.models import Sequential, Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
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
                img = Image.open(f).convert("RGB")  # ép tất cả về RGB (3 kênh)
                img = img.resize((28, 28))
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
                img = img.resize((28, 28))
                X_test.append(np.array(img))
                y_test.append(class_idx)
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Kích thước X_test:", X_test.shape)
print("Kích thước y_test:", y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_new = to_categorical(y_train, num_classes = 21)
y_val_new = to_categorical(y_val, num_classes = 21)

model_LeNet = Sequential([
    Conv2D(6, (5, 5), activation='relu', padding= 'same', input_shape=(28, 28, 3)),
    AveragePooling2D(strides=2, pool_size = (2, 2)),
    
    Conv2D(16, (5, 5), activation='relu', padding='valid'),
    AveragePooling2D(strides=2, pool_size = (2, 2)),

    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation = 'relu'),
    Dense(21, activation='softmax')
])

Optimizer = Adam(learning_rate=1e-3)
Loss = CategoricalCrossentropy()
model_LeNet.compile(optimizer=Optimizer, loss=Loss, metrics=['accuracy'])

history_1 = model_LeNet.fit(X_train, y_train_new, 
                            validation_data=(X_val, y_val_new),
                            batch_size=32, epochs=20)


y_pred = model_LeNet.predict(X_test)
y_pred = np.argmax(y_pred, axis = -1)

accuracy = round(accuracy_score(y_test, y_pred)*100,2)
precision = round(precision_score(y_test, y_pred, average='macro')*100,2)
recall = round(recall_score(y_test, y_pred, average='macro')*100,2)
f1 = round(f1_score(y_test, y_pred, average='macro')*100,2)

print('Accuracy test = {}%'.format(accuracy))
print('Precision test = {}%'.format(precision))
print('Recall test = {}%'.format(recall))
print('F1-macro test = {}%'.format(f1))
