import keras
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.python.keras import Sequential

from data_loading import signs

model: Sequential = keras.models.load_model('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\cnn_model_1.h5')

# Load test data from source
test_data, test_labels = signs.test_data_greyscale()

# Predict classes from test data and get predicted classes
pred = np.argmax(model.predict(test_data), axis=-1)

# Check accuracy with the test data
print(classification_report(test_labels, pred))

model.summary()
print(model.get_config()['layers'])

