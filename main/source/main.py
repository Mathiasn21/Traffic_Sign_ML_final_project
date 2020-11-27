# TODO: Process data
import mnist
import numpy as np
# TODO: Analyze testing
from numpy.core.multiarray import ndarray

from ml_models.cnn.layer.dense import Dense

# TODO: Setup model and add layers.
# TODO: Train data
# TODO: Test data

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

softmax = Dense(28 * 28 * 1, 10, activation='softmax')


def forward(image, label):
    """
    Completes a forward pass of the cnn and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    """
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.

    out = softmax.forward_propagate((image / 255) - 0.5)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=.005):
    """
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    """
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.back_propagate(gradient)
    # TODO: backprop MaxPool2 layer
    # TODO: backprop Conv3x3 layer

    return loss, acc


print('MNIST CNN initialized!')
# Train!
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
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
