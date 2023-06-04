# -*- coding: utf-8 -*-

import os
import pandas as pd 
import numpy as np 
import random

from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import models, layers

import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import cv2
import matplotlib.image as mpimg

from matplotlib import pyplot as plt
# %matplotlib inline

path = "data/gym"
path_imgs = path + "IMG/"
df = pd.read_csv(path + "/driving_log.csv",  names=['center','left','right','steering','throttle','brake','speed'])
df.head()

# PRÉ-PROCESSAMENTO
def get_image_name(file_path):
    # On Windows: filePath.split("\\")[-1]
    # On Linux: filePath.split("/")[-1]

    return file_path.split("/")[-1]

columns = ['center','left','right','steering','throttle','brake','speed']
df = pd.read_csv(os.path.join(path, "driving_log.csv"),  names=columns)
df['center'] = df['center'].apply(get_image_name)
df['left'] = df['left'].apply(get_image_name)
df['right'] = df['right'].apply(get_image_name)

print(df['center'][0])
print(f'Total imported images : {df.shape[0]}')

### BALANCEAMENTO DOS DADOS
nbins = 31
# Set data pico
samples_per_bin = 300

hist, bins = np.histogram(df['steering'], nbins)
print(bins)

center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.06)
plt.plot((-1,1), (samples_per_bin, samples_per_bin))
plt.show()

### EXCLUINDO DADOS DESNECESSÁRIOS
remove_index_list = []
for j in range(nbins):
    bin_data_list = []

    for i in range(len(df['steering'])):
        if df['steering'][i] >= bins[j] and df['steering'][i] <= bins[j + 1]:
            bin_data_list.append(i)

    bin_data_list = shuffle(bin_data_list)
    bin_data_list = bin_data_list[samples_per_bin:]

    remove_index_list.extend(bin_data_list)

print('Removed Images:', len(remove_index_list))
df.drop(df.index[remove_index_list], inplace=True)
print('Remaining Images:', len(df))

hist, _ = np.histogram(df['steering'], (nbins))
plt.bar(center, hist, width=0.06)
plt.plot((np.min(df['steering']), np.max(df['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

def load_data(path, data):
    images_path = []
    steering = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        choice = np.random.choice(3)

        if choice == 0:
            images_path.append(os.path.join(path, 'IMG' ,indexed_data[0]))
            steering.append(float(indexed_data[3]))

        elif choice == 1:
            images_path.append(os.path.join(path, 'IMG' ,indexed_data[1]))
            steering.append(float(indexed_data[3])+0.2)

        else:
            images_path.append(os.path.join(path, 'IMG' ,indexed_data[1]))
            steering.append(float(indexed_data[3])-0.2)

    images_path = np.asarray(images_path)
    steering = np.asarray(steering)

    return images_path, steering

x, y = load_data(path, df)

### CRIANDO CONJUTOS DE TREINO E TESTE
xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

### DATA AUGMENTATION
def pre_process(img):
    img = img[60:135,:,:] #CROP
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #RGB TO YUV
    img = cv2.GaussianBlur(img,  (3, 3), 0) # BLUR
    img = cv2.resize(img, (200, 66)) #RESIZE
    img = img/255

    return img

def augment_image(imgPath,steering):
    img = mpimg.imread(imgPath)

    #PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    #BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)

    #FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering

def batch_gen(images_path, steering_list, batch_size, train_flag):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0, len(images_path) - 1)

            if train_flag: #APLICA DATA AUG NO TREINO
                img, steering = augment_image(images_path[index], steering_list[index])

            else: # CARREGA IMAGEM NA VALIDAÇÃO
                img = mpimg.imread(images_path[index])
                steering = steering_list[index]

            img = pre_process(img)
            img_batch.append(img)
            steering_batch.append(steering)

        yield (np.asarray(images_path), np.asarray(steering_batch))

### HIPERPARÂMETROS
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

EPOCHS = 50
BATCH_SIZE = 64
alpha  = 1e-5

### CONVOLUTIONAL NEURAL NETWORK
model = Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='elu', padding="same", input_shape=INPUT_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128,(3, 3), activation='elu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, (3, 3), activation='elu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, (3, 3), activation='elu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(512, (1, 1), activation='elu', padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(258, activation="elu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae', 'mse'])

### KERAS CALLBACKS
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)

stopping = EarlyStopping(monitor="val_loss", min_delta=alpha, patience=15, verbose=1)

callbacks = [checkpoint, lr_reduce, stopping]

### TREINAMENTO
history = model.fit(batch_gen(xTrain, yTrain,batchSize=100, trainFlag=1),
                    steps_per_epoch=300,
                    epochs=50,
                    validation_data=batch_gen(xVal, yVal, batchSize=100, trainFlag=0),
                    validation_steps=200,
                    callbacks = callbacks
                    )

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model error')
plt.ylabel('erro')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

