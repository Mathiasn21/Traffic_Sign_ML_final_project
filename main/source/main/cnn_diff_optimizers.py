import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loading import signs

# Load training data from source
data, labels = signs.training_data_grayscale()
classes = 43  # Total number of traffic sign classes

# Splitting data into training and test data. Does shuffle before split in order to increase randomness in the data.
# Specifying the random state allows for reproducibility.
x_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Function to build and compile CNN models with varying optimizers
def build_models_diff_optimizers():
    # Array consisting of various optimizers
    optimizer_arr = ['adam', 'sgd', 'rmsprop', 'adadelta']

    # Array consisting of various optimizer names
    opt_names = ['Adam', 'SGD', 'RMSprop', 'Adadelta']
    model_array = []  # Array utilized for storing compiled CNN models

    # Build and compile CNN models, varying the optimizer
    for optimizer in optimizer_arr:
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(.25))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(.5))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model_array.append(model)
    return model_array, opt_names


def train_models(models, opt_names):
    """
    Function to train CNN models, where each conv2d layer has varying activation functions
    :param opt_names: object - Array consisting of optimizer names
    :param models: object - Array consisting of CNN models.
    """
    epochs = 16

    # Load test data from source
    test_data, test_labels = signs.test_data_greyscale()

    # Array utilized for storing histories gotten from fitting a CNN model.
    histories = []

    for i, model in enumerate(models):
        # Fit the CNN model using training data and labels.
        history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
        pred = np.argmax(model.predict(test_data), axis=-1)  # Select highest probability for all classifications
        histories.append(history)  # Append new history stats to the histories array

        # Check accuracy with the test data
        print(accuracy_score(test_labels, pred))

    plt.figure(0)
    for i, hist in enumerate(histories):
        # plotting graphs of accuracy from various optimizers
        plt.plot(hist.history['accuracy'], label=opt_names[i])

    plt.title('Accuracy - Optimizers')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    for i, hist in enumerate(histories):
        # plotting graphs of loss from various optimizers
        plt.plot(hist.history['loss'], label=opt_names[i])
    plt.title('Loss - Optimizers')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


# Build CNN models then train and display data from them.
models, opt_names = build_models_diff_optimizers()
train_models(models, opt_names)
