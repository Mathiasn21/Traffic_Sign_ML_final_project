import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from data_loading import signs

data, labels = signs.training_data()
classes = 43

# Splitting data into training and test data. Does shuffle before split in order to increase randomness in the data.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=21)

# Convert labels into binary class matrix
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Build CNN model
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

# Compile CNN
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
epochs = 16
history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test))

# Save model onto disk
model.save("D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_7.h5")

# plotting graphs of accuracy and loss
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy - ReLu')
plt.plot(history.history['val_accuracy'], label='val accuracy  - ReLu')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss - ReLu')
plt.plot(history.history['val_loss'], label='val loss - ReLu')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# model = keras.models.load_model('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_6.h5')

# Load test data from source
test_data, test_labels = signs.test_data()

# Predict classes from test data and get predicted classes
pred = np.argmax(model.predict(test_data), axis=-1)

# Check accuracy with the test data
print(classification_report(test_labels, pred))

