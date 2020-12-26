import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from data_loading import signs
import matplotlib.pyplot as plt

model: Sequential = keras.models.load_model(
    'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_7.h5')

# Load test data from source
test_data, test_labels = signs.test_data_greyscale()

# Predict classes from test data and get predicted classes
pred = np.argmax(model.predict(test_data), axis=-1)
desired_width = 620
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
pd.set_option('display.width', desired_width)
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
predictions = mlp.predict(x_test)
print(classification_report(test_labels, predictions))
print(pd.DataFrame(confusion_matrix(test_labels, predictions)))

fig, ax = plt.subplots(figsize=(15, 15))
disp = plot_confusion_matrix(mlp, x_test, test_labels, ax=ax)
disp.ax_.set_title('Confusion Matrix')
plt.show()
