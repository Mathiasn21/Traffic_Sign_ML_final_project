import os
import numpy as np
import pandas as pd

from PIL import Image

training_images = []
training_labels = []
test_images = []
test_labels = []

classes = 43
data_path = "C:\\Users\\Nilse\\Desktop\\archive"


def training_data_grayscale():
    # Retrieving the images and their labels
    for i in range(classes):
        path = os.path.join(data_path, 'Train', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                # Open image location and convert to grayscale
                image: Image = Image.open(path + '\\' + a).convert('LA')
                image = image.resize((30, 30))
                image = np.array(image)
                training_images.append(image)
                training_labels.append(i)
            except:
                print("Error loading image")

    return np.array(training_images), np.array(training_labels)


def training_data():
    # Retrieving the images and their labels
    for i in range(classes):
        path = os.path.join(data_path, 'Train', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                # Open image location
                image: Image = Image.open(path + '\\' + a)
                image = image.resize((30, 30))
                image = np.array(image)
                training_images.append(image)
                training_labels.append(i)
            except:
                print("Error loading image")
    return np.array(training_images), np.array(training_labels)


def test_data_greyscale():
    y_test = pd.read_csv('C:\\Users\\Nilse\\Desktop\\archive\\Test.csv')
    labels = y_test["ClassId"].values
    images = "C:\\Users\\Nilse\\Desktop\\archive\\" + y_test["Path"].values
    for img in images:
        image = Image.open(img).convert('LA')
        image = image.resize((30, 30))
        image = np.array(image)

        test_images.append(np.array(image))
    return np.array(test_images), np.array(labels)


def test_data():
    y_test = pd.read_csv('C:\\Users\\Nilse\\Desktop\\archive\\Test.csv')
    labels = y_test["ClassId"].values
    images = "C:\\Users\\Nilse\\Desktop\\archive\\" + y_test["Path"].values

    for img in images:
        image = Image.open(img)
        image = image.resize((30, 30))
        image = np.array(image)

        test_images.append(np.array(image))
    return np.array(test_images), np.array(labels)
