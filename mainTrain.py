import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load and preprocess data
=======
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical

>>>>>>> 5c45c0898a4bea319c4321d6561d2bfdb70b4d8d
image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'non_demented/')
yes_tumor_images = os.listdir(image_directory + 'demented/')
dataset = []
label = []

INPUT_SIZE = 64

<<<<<<< HEAD
for image_name in no_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'non_demented/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(image)
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'demented/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(image)
        label.append(1)

dataset = np.array(dataset) / 255.0
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
=======
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'non_demented/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'demented/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize pixel values using MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, INPUT_SIZE * INPUT_SIZE * 3)).reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)
x_test = scaler.transform(x_test.reshape(-1, INPUT_SIZE * INPUT_SIZE * 3)).reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)
>>>>>>> 5c45c0898a4bea319c4321d6561d2bfdb70b4d8d

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Model Building
model = Sequential()

<<<<<<< HEAD
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
=======
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
>>>>>>> 5c45c0898a4bea319c4321d6561d2bfdb70b4d8d
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
<<<<<<< HEAD
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))

# Save the model
model.save('alzheimer_detection_model.h5')
=======
model.add(Dense(2))
model.add(Activation('softmax'))

# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=16,
          verbose=1,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=False)

model.save('Alzheimer10EpochsCategorical.h5')
>>>>>>> 5c45c0898a4bea319c4321d6561d2bfdb70b4d8d
