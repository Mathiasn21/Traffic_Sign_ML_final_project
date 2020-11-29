import mnist

from ml_models.cnn.cnn import CNN
from ml_models.cnn.layer.dense import Dense

training_images = mnist.test_images()[:1000]
training_labels = mnist.test_labels()[:1000]
num_labels = 10

cnn = CNN(1, training_images, training_labels)
cnn.add(Dense(num_labels, 'softmax'))
cnn.fit()
