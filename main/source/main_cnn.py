import joblib
import mnist

from ml_models.cnn.cnn import CNN
from ml_models.cnn.layer.conv2d import Conv2D
from ml_models.cnn.layer.dense import Dense
from ml_models.cnn.layer.maxpool2d import MaxPool2d

training_images = mnist.train_images()[:10000]
training_labels = mnist.train_labels()[:10000]

training_images = (training_images / 255) - 0.5

num_labels = 10

cnn = CNN(5, training_images, training_labels)
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(MaxPool2d())
cnn.add(Dense(num_labels, 'softmax'))
cnn.fit()

joblib.dump(cnn, 'cnn_mnist.joblib')
best_cnn = joblib.load('cnn_mnist.joblib')

print("")
