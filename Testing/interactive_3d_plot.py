from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ModelResult:
    def __init__(self, info_dict: dict):
        self.model = info_dict['model']
        self.arg = info_dict['arg']
        self.predictors = info_dict['predictors']
        self.predicted = info_dict['predicted']
        self.error = info_dict['error']


def split_by_threshold(ss, predictor, threshold):
    return ss[ss[predictor] < threshold], ss[ss[predictor] >= threshold]


def polynomial_regression_cv(x, y, arg=None, inputs=None):
    poly_features = PolynomialFeatures(degree=arg, include_bias=False).fit_transform(x)
    cv_scores = cross_val_score(estimator=LinearRegression(),
                                X=poly_features,
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=poly_features,
                                       y=y,
                                       cv=5)
    return ModelResult({'model': 'polynomial_regression',
                        'arg': f'order={arg}',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum() / len(cv_scores)})


def multiple_linear_regression_cv(x, y, arg=None, inputs=None):
    cv_scores = cross_val_score(estimator=LinearRegression(),
                                X=x[inputs],
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=x[inputs],
                                       y=y,
                                       cv=5)
    return ModelResult({'model': 'multiple_linear_regression',
                        'arg': '',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum() / len(cv_scores)})


def inputs_cv():
    predictors = ['x1', 'x2', 'x3', 'x4', 'x5']

    predictor_combinations = []
    for L in range(2, len(predictors) + 1):
        predictor_combinations.extend(combinations(predictors, L))

    inp_combs = list(map(list, predictor_combinations))

    for inp in inp_combs:
        target_model = polynomial_regression_cv(x=x3_data[inp],
                                                y=x3_data['Y'],
                                                inputs=inp, arg=2)
        comp = pd.DataFrame({'Predicted': target_model.predicted, 'Actual': x3_data['Y']})
        comp['Error'] = (comp['Predicted'] - comp['Actual']).abs()
        # print(comp)
        print(f'P = {inp}', target_model.error)


np.set_printoptions(precision=3, suppress=True)
ecm = make_scorer(mean_absolute_error)

raw_dataset = pd.read_csv('../dataset.csv', sep=',', skipinitialspace=True)
dataset = raw_dataset.copy()
predict_set = dataset.tail(20)
dataset = dataset.drop(predict_set.index).drop('SampleNo', axis=1).drop('x6', axis=1)

# Split by threshold
tr_1_lt, tr_1_gt = split_by_threshold(dataset, 'x5', threshold=5)
tr_2_lt, tr_2_gt = split_by_threshold(tr_1_lt, 'x1', threshold=20)
x5_data = tr_1_gt
x1_data = tr_2_lt
x3_data = tr_2_gt

target_model = polynomial_regression_cv(x=x3_data[['x3', 'x1']],
                                        y=x3_data['Y'],
                                        inputs=['x3', 'x1'], arg=2)

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(elev=2., azim=-100)
ax.set_box_aspect(aspect=None, zoom=1.35)

ax.scatter(x3_data['x3'], x3_data['x1'], x3_data['Y'], c=x3_data['Y'])
ax.plot_trisurf(x3_data['x3'], x3_data['x1'], target_model.predicted,
                linewidth=0, antialiased=True, alpha=0.3, color='r')

ax.set_xlabel('x3')
ax.set_ylabel('x1')
ax.set_zlabel('Y')
plt.show()

# predictions = predictions.sort_values('SampleNo')
# with pd.option_context('display.max_rows', None,):
#     display(predictions)
