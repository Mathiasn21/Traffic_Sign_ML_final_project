import os

import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from ml_models.cnn.cnn import CNN
from ml_models.cnn.layer.conv2d import Conv2D
from ml_models.cnn.layer.dense import Dense
from ml_models.cnn.layer.maxpool2d import MaxPool2d

num_labels = 43
data = []
labels = []
classes = 43
data_path = "C:\\Users\\Nilse\\Desktop\\archive"

# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(data_path, 'Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\' + a).convert('LA')
            image = image.resize((30, 30))
            image = np.array(image)[:, :, 0] / 255 - 0.5
            # sim = Image.from array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

cnn = CNN(5, X_test, y_test)
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(MaxPool2d())
cnn.add(Dense(num_labels, 'softmax'))
cnn.fit()

joblib.dump(cnn, 'cnn_sign.joblib')
best_cnn = joblib.load('cnn_sign.joblib')
