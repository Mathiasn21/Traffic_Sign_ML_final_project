import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from ml_models.cnn.cnn import CNN

y_test = pd.read_csv('C:\\Users\\Nilse\\Desktop\\archive\\Test.csv')

labels = y_test["ClassId"].values
imgs = "C:\\Users\\Nilse\\Desktop\\archive\\" + y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img).convert('LA')
    image = image.resize((30, 30))
    image = np.array(image)[:, :, 0] / 255 - 0.5

    data.append(np.array(image))

test_images = np.array(data)

signs_cnn: CNN = joblib.load('cnn_sign.joblib')
pred = signs_cnn.predict(test_images)

print(accuracy_score(labels, pred))
print(classification_report(labels, pred))
