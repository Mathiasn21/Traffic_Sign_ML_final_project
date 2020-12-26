import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loading import signs

data, labels = signs.training_data_grayscale()
classes = 43

# Splitting data into training and test data. Does shuffle before split in order to increase randomness in the data.
x_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels into one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)


def build_models_diff_optimizers():
    optimizer_arr = ['adam', 'sgd', 'rmsprop', 'adadelta']
    opt_names = ['Adam', 'SGD', 'RMSprop', 'Adadelta']
    model_array = []

    for optimizer in optimizer_arr:
        # Build CNN models
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
    epochs = 16
    test_data, test_labels = signs.test_data_greyscale()
    histories = []

    for i, model in enumerate(models):
        history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
        pred = np.argmax(model.predict(test_data), axis=-1)
        histories.append(history)

        # Check accuracy with the test data
        print(accuracy_score(test_labels, pred))

    plt.figure(0)
    for i, hist in enumerate(histories):
        # plotting graphs of accuracy and loss
        plt.plot(hist.history['accuracy'], label=opt_names[i])

    plt.title('Accuracy - Optimizers')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    for i, hist in enumerate(histories):
        plt.plot(hist.history['loss'], label=opt_names[i])
    plt.title('Loss - Optimizers')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


models, opt_names = build_models_diff_optimizers()
train_models(models, opt_names)