import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_loading import signs

x_train, y_train = signs.training_data_grayscale()
x_train = x_train[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_train)
x_train = x_train.reshape((n, -1))

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)

mlp = MLPClassifier(solver='adam',
                    hidden_layer_sizes=(150, 110, 43),
                    max_iter=1000,
                    learning_rate_init=.0001,
                    activation='relu')

mlp.fit(x_train, y_train.ravel())

joblib.dump(mlp, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_3.joblib')
