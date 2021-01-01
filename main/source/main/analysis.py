import joblib
import keras
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Sequential
from data_loading import signs

# Set settings to ensure no truncation when printing the dataframe, from confusion matrix
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 300)

# Load stored keras model - Should probably change this such that it matches your directory
model: Sequential = keras.models.load_model(
    'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_best.h5')

# Load test data from source
test_data, test_labels = signs.test_data_greyscale()

# Predict classes from test data and get predicted classes
pred = np.argmax(model.predict(test_data), axis=-1)

# Print model summary - Details pertaining to each layer as well as total params.
print("\n\n\t\tPrinting model summary(CNN):\n")
model.summary()
print("\n\n")

# Check accuracy with the test data
print("\n\n\t\tPrinting CNN classification report:\n")
print(classification_report(test_labels, pred))
print("\n")

# Create a dataframe which makes it easier to pretty print the confusion matrix for CNN model
print("\n\n\t\tPrinting CNN confusion matrix:\n")
print(pd.DataFrame(confusion_matrix(test_labels, pred)))
print("\n")

# Create a scalar used for scaling the image values
scalar = StandardScaler()

# Selects only the grey column for each x => Decreased training time and less data
x_test = test_data[:, :, :, 0]

n = len(x_test)  # Total images
# Flatten image values
x_test = x_test.reshape((n, -1))

# Computes mean and std - used later for scaling the dataset
scalar.fit(x_test)
# Scale the dataset by centering and scaling
x_test = scalar.transform(x_test)

# Load pre saved best MLP model
mlp: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_adam_best.joblib')

# Try to predict traffic sign given test data
predictions = mlp.predict(x_test)

# Print classification report
print("\n\n\t\tPrinting MLP classification report:\n")
print(classification_report(test_labels, predictions))
print("\n")

# Create a dataframe which makes it easier to pretty print the confusion matrix for MLP model
print("\n\n\t\tPrinting MLP confusion matrix:\n")
print(pd.DataFrame(confusion_matrix(test_labels, predictions)))
