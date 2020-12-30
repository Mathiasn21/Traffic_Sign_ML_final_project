import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential

from data_loading import signs

model: Sequential = keras.models.load_model(
    'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_7.h5')

# Load test data from source
test_data, test_labels = signs.test_data_greyscale()

# Predict classes from test data and get predicted classes
pred = np.argmax(model.predict(test_data), axis=-1)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)

# Check accuracy with the test data
print(classification_report(test_labels, pred))
print(pd.DataFrame(confusion_matrix(test_labels, pred)))

model.summary()
scalar = StandardScaler()

x_test = test_data[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_test)
x_test = x_test.reshape((n, -1))

scalar.fit(x_test)
x_test = scalar.transform(x_test)

mlp: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_1.joblib')
mlp2: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_2.joblib')
mlp3: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_3.joblib')
predictions = mlp.predict(x_test)

print(classification_report(test_labels, predictions))
print(pd.DataFrame(confusion_matrix(test_labels, predictions)))

plt.figure(0)
plt.plot(mlp.loss_curve_, label='Adam - Learning rate 0.0001')
plt.plot(mlp2.loss_curve_, label='SGD - Learning rate 0.01')
plt.plot(mlp3.loss_curve_, label='Adam - Learning rate 0.01')
plt.title('Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
