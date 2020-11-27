from numpy.core.multiarray import ndarray
import numpy as np

from ml_models.cnn.layer.dense import Dense


class CNN:
    epochs: int
    learning_rate: float
    training_features: ndarray
    training_labels: ndarray

    def __init__(self,
                 epochs: int,
                 training_features: ndarray,
                 training_labels: ndarray,
                 learning_rate: float = .005):
        self.training_features = training_features
        self.training_labels = training_labels
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, image, label):
        """
        Completes a forward pass of the cnn and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        """
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.

        softmax = Dense(26 * 26 * 1, 10, activation='softmax')
        out = softmax.forward_propagate(ndarray((image / 255) - 0.5))

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train(self, im, label, lr=.005):

        """
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        """
        # Forward
        out, loss, acc = self.forward(im, label)

        # Calculate initial gradient
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # Backprop
        gradient = softmax.backprop(gradient, lr)
        # TODO: backprop MaxPool2 layer
        # TODO: backprop Conv3x3 layer

        return loss, acc

    def fit(self):
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(self.training_features, self.training_labels)):
            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0

            l, acc = train(im, label)
            loss += l
            num_correct += acc

    def test(self, test_features: ndarray, test_labels: ndarray):
        print("")
        # TODO: implement method

