import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from data_loading import signs


def search_params(train_x, train_y):
    c = [2.0, 2.2, 2.4, 2.8]
    gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    parameters = {'kernel': ['rbf'], 'gamma': gamma, 'C': c}
    # best so far: C = 2.0, gamma = 0.00101

    model = SVC()
    grid_search = GridSearchCV(model, parameters, refit=True, n_jobs=4, verbose=1)
    grid_search.fit(train_x, train_y)
    return grid_search.best_params_


x_features, y_targets = signs.training_data_grayscale()
x_features = x_features[:, :, :, 0]
n = len(x_features)
x_features = x_features.reshape((n, -1))

x_train, x_test, y_train, y_test = train_test_split(x_features, y_targets, test_size=0.2)

best_params = search_params(x_train, y_train)
best_svc = SVC(C=2.0, gamma=0.00001, kernel='rbf', verbose=True)
best_svc.fit(x_train, y_train)
svc_predictions = best_svc.predict(x_test)

print(best_params)

joblib.dump(best_svc, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\svm_1.joblib')
best_svc = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\svm_1.joblib')

print('Classification report: ', classification_report(y_test, svc_predictions), '\n')
