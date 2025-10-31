import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn

import os
from PIL import Image

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Input, MaxPooling2D, Conv2D, BatchNormalization, \
    Flatten, Dropout, AveragePooling2D, Rescaling, concatenate, GlobalAveragePooling2D
from keras.models import Sequential, Model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.models import load_model
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

def inception_block(x, filters):
    f1, f3In, f3O, f5In, f5O, fp = filters

    branch1 = Conv2D(f1, (1,1), padding='same', activation='relu')(x)

    branch2 = Conv2D(f3In, (1,1), padding='same', activation='relu')(x)
    branch2 = Conv2D(f3O, (3,3), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(f5In, (1,1), padding='same', activation='relu')(x)
    branch3 = Conv2D(f5O, (5,5), padding='same', activation='relu')(branch3)

    branch4 = MaxPooling2D((3,3), strides=1, padding='same')(x)
    branch4 = Conv2D(fp, (1,1), padding='same', activation='relu')(branch4)

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return x


def myGoogLeNet(input_shape=(224,224,3), num_classes=21):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7,7), strides=2, padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (1,1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_block(x, (64, 96, 128, 16, 32, 32))   # 3a
    x = inception_block(x, (128, 128, 192, 32, 96, 64)) # 3b
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_block(x, (192, 96, 208, 16, 48, 64))  # 4a
    x = inception_block(x, (160, 112, 224, 24, 64, 64)) # 4b
    x = inception_block(x, (128, 128, 256, 24, 64, 64)) # 4c
    x = inception_block(x, (112, 144, 288, 32, 64, 64)) # 4d
    x = inception_block(x, (256, 160, 320, 32, 128, 128)) # 4e
    x = MaxPooling2D((3,3), strides=2, padding='same')(x)

    x = inception_block(x, (256, 160, 320, 32, 128, 128)) # 5a
    x = inception_block(x, (384, 192, 384, 48, 128, 128)) # 5b

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='GoogLeNet_v1')
    return model

model_googlenet = myGoogLeNet(input_shape=(224, 224, 3), num_classes=21)

model_googlenet.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

history_2 = model_googlenet.fit(X_train, y_train_new, 
                                validation_data=(X_val, y_val_new),
                                batch_size=64, epochs=20)

y_pred = model_googlenet.predict(X_test)
y_pred = np.argmax(y_pred, axis = -1)

accuracy = round(accuracy_score(y_test, y_pred)*100,2)
precision = round(precision_score(y_test, y_pred, average='macro')*100,2)
recall = round(recall_score(y_test, y_pred, average='macro')*100,2)
f1 = round(f1_score(y_test, y_pred, average='macro')*100,2)

print('Accuracy test = {}%'.format(accuracy))
print('Precision test = {}%'.format(precision))
print('Recall test = {}%'.format(recall))
print('F1-macro test = {}%'.format(f1))