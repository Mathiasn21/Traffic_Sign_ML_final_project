import joblib
import mnist
from sklearn.metrics import accuracy_score, classification_report

from ml_models.cnn.cnn import CNN
from ml_models.cnn.layer.conv2d import Conv2D
from ml_models.cnn.layer.dense import Dense
from ml_models.cnn.layer.maxpool2d import MaxPool2d

training_images = mnist.train_images()[:10000]
training_labels = mnist.train_labels()[:10000]
training_images = (training_images / 255) - 0.5
num_labels = 10


def train_cnn():
    cnn = CNN(8, training_images, training_labels)
    cnn.add(Conv2D(16, (3, 3), activation='relu'))
    cnn.add(MaxPool2d())
    cnn.add(Dense(num_labels, 'softmax'))
    cnn.fit()
    joblib.dump(cnn, '../source/cnn_mnist2.joblib')


def test_cnn(cnn):
    test_images = (mnist.test_images() / 255) - 0.5
    test_labels = mnist.test_labels()

    predictions = cnn.predict(test_images)
    return {'pred': predictions, 'test_labels': test_labels}


mnist_cnn = joblib.load('../source/cnn_mnist.joblib')
mnist_cnn2 = joblib.load('../source/cnn_mnist2.joblib')

res = test_cnn(mnist_cnn)
res2 = test_cnn(mnist_cnn2)

print(accuracy_score(res['test_labels'], res['pred']))
print(classification_report(res['test_labels'], res['pred']))
print(accuracy_score(res2['test_labels'], res2['pred']))
print(classification_report(res2['test_labels'], res2['pred']))
