import time

import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from data_loading import signs

x_train, y_train = signs.training_data_grayscale()
x_train = x_train[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_train)
x_train = x_train.reshape((n, -1))

x_test, y_test = signs.test_data_greyscale()
x_test = x_test[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_test)
x_test = x_test.reshape((n, -1))

start_time = time.time()
best_svc = SVC(C=2.1, gamma=0.0001, kernel='rbf')
best_svc.fit(x_train, y_train)
svc_predictions = best_svc.predict(x_test)

joblib.dump(best_svc, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_1.joblib')
# best_mlp = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\.joblib')

print('Classification report: ', classification_report(y_test, svc_predictions), '\n')
print('End time: ', time.time() - start_time, '\n')

conf_matrix = confusion_matrix(y_test, svc_predictions)
