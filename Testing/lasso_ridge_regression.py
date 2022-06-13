from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pd.set_option("display.precision", 5)
np.set_printoptions(precision=5, suppress=True)

# Dataset
raw_dataset = pd.read_csv('../dataset.csv', sep=',', skipinitialspace=True)
dataset = raw_dataset.copy()
predict_set = dataset.tail(20)
dataset = dataset.drop(predict_set.index)

predictors = ['x1', 'x2', 'x3', 'x4', 'x5']
features = pd.DataFrame(StandardScaler().fit_transform(dataset[predictors]), columns=predictors)
labels = dataset['Y']


kf = KFold(n_splits=5, random_state=42, shuffle=True)

cv_scores = []
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    alphas = np.linspace(0.1, 2000, 5000)
    params = {'alpha': alphas}

    clf = GridSearchCV(Lasso(), params, cv=5, scoring=make_scorer(mean_absolute_error))
    clf.fit(X_train[predictors], y_train)

    print(predictors, clf.best_params_['alpha'], clf.best_score_)
