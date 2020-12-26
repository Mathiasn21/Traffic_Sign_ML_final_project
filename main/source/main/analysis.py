import keras
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from data_loading import signs

model: Sequential = keras.models.load_model('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_7.h5')

# Load test data from source
test_data, test_labels = signs.test_data_greyscale()

# Predict classes from test data and get predicted classes
pred = np.argmax(model.predict(test_data), axis=-1)

# Check accuracy with the test data
print(classification_report(test_labels, pred))

model.summary()

x_train, y_train = signs.training_data_grayscale()
x_train = x_train[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_train)
x_train = x_train.reshape((n, -1))

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)

x_test = test_data[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_test)
x_test = x_test.reshape((n, -1))

scalar.fit(x_test)
x_test = scalar.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(150, 110, 43), max_iter=1000, learning_rate_init=.0001, activation='relu')
mlp.fit(x_train, y_train.ravel())
predictions = mlp.predict(x_test)
print(classification_report(test_labels, predictions))
