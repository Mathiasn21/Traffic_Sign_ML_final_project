import mnist

from ml_models.cnn.cnn import CNN
from ml_models.cnn.layer.conv2d import Conv2D
from ml_models.cnn.layer.dense import Dense
from ml_models.cnn.layer.maxpool2d import MaxPool2d

training_images = mnist.train_images()[:1000]
training_labels = mnist.train_labels()[:1000]
num_labels = 10

cnn = CNN(4, training_images, training_labels)
cnn.add(Conv2D(8, (3, 3), activation='relu'))
cnn.add(MaxPool2d())
cnn.add(Dense(num_labels, 'softmax'))
cnn.fit()
