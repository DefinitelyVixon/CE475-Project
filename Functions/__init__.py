from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


np.set_printoptions(precision=3, suppress=True)
ecm = make_scorer(mean_absolute_error)
ecm_name = 'mean_absolute_error'
k_fold_value = 5

raw_dataset = pd.read_csv('dataset.csv', sep=',', skipinitialspace=True)
dataset = raw_dataset.copy()
predict_set = dataset.tail(20)
dataset = dataset.drop(predict_set.index)

predictors = ['SampleNo', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']


# Utility Classes
class ModelResult:
    def __init__(self, info_dict: dict):
        self.model = info_dict['model']
        self.arg = info_dict['arg']
        self.predictors = info_dict['predictors']
        self.predicted = info_dict['predicted']
        self.error = info_dict['error']
        if 'estimator' in info_dict.keys():
            self.estimator = info_dict['estimator']


class ModelResultsTable:
    def __init__(self):
        self.all_model_results = {}  # {'mul_lin_reg': [ModelResult0, ModelResult1, ...], 'lin_reg': [...] ... }
        self.model_results_with_best_params = []

    def add_model(self, model_results: list):
        sorted_models = list(sorted(model_results, key=lambda item: item.error))
        best_performing_var = sorted_models[0]
        self.all_model_results[best_performing_var.model] = sorted_models
        self.model_results_with_best_params.append(best_performing_var)
        self.model_results_with_best_params = list(sorted(self.model_results_with_best_params,
                                                          key=lambda item: item.error))

    def display_error_table(self, transpose=True, model=None):
        if model is not None:
            arg_indexes = [model_result.arg for model_result in self.all_model_results[model]]
            error_table = pd.DataFrame([[result.predictors, result.error]
                                        for result in self.all_model_results[model]],
                                       columns=['predictors', ecm_name], index=arg_indexes)
        else:
            model_indexes = [model_result.model for model_result in self.model_results_with_best_params]
            error_table = pd.DataFrame([[result.arg, result.predictors, result.error]
                                        for result in self.model_results_with_best_params],
                                       columns=['args', 'predictors', ecm_name], index=model_indexes)
        if transpose:
            error_table = error_table.transpose()
        display(error_table)


# Utility Functions
def plot_conditional_models(cond_models, cond_sets, plot_titles):
    fig_ = plt.figure(figsize=(18, 5))
    cv_error = 0
    plot_i = 1
    decision_df = pd.DataFrame(columns=['Model', 'Predictors', 'Args', 'Error'])

    for cond_model, cond_set, plot_title in zip(cond_models, cond_sets, plot_titles):
        decision_df.loc[plot_title] = [cond_model.model, cond_model.predictors,
                                       cond_model.arg, cond_model.error]

        if len(cond_model.predictors) == 1:
            ax = fig_.add_subplot(1, 3, plot_i)
            sns.scatterplot(ax=ax, x=cond_set[cond_model.predictors[0]], y=cond_set['Y'],
                            color='b', label='Actual Data')
            sns.scatterplot(ax=ax, x=cond_set[cond_model.predictors[0]], y=cond_model.predicted,
                            color='r', label='Predicted')
            ax.set_title(plot_title)
        elif len(cond_model.predictors) == 2:
            ax = fig_.add_subplot(1, 3, plot_i, projection='3d')
            ax.view_init(elev=2., azim=-96)
            ax.set_box_aspect(aspect=None, zoom=1.35)

            x_axis = cond_model.predictors[0]
            y_axis = cond_model.predictors[1]
            z_axis = 'Y'

            ax.scatter(cond_set[x_axis],
                       cond_set[y_axis],
                       cond_set[z_axis],
                       c=cond_set[z_axis])
            ax.plot_trisurf(cond_set[x_axis],
                            cond_set[y_axis],
                            cond_model.predicted,
                            linewidth=0, antialiased=True, alpha=0.3, color='r')
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_zlabel(z_axis)
            ax.set_title(plot_title)
        else:
            pass

        cv_error += cond_model.error
        plot_i += 1

    print('Total CV Score (MAE) =', cv_error)
    display(decision_df)


def subplot_for_five(data, title=None, highlight=None, reg_plot=None):
    point_colors = ['b', 'b', 'b', 'b', 'b']
    if highlight is not None:
        point_colors[highlight] = 'r'
    fig_, axes_ = plt.subplots(2, 3, figsize=(18, 10))
    fig_.suptitle(title, fontsize=20)

    axes_[1][2].set_visible(False)
    axes_[1][0].set_position([0.24, 0.125, 0.228, 0.343])
    axes_[1][1].set_position([0.55, 0.125, 0.228, 0.343])

    if reg_plot is not None:
        for i in range(len(predictors)):
            sns.scatterplot(ax=axes_[i // 3, i % 3], x=data[predictors[i]], y=data['Y'],
                            alpha=0.5, color=point_colors[i])
            sns.regplot(ax=axes_[i // 3, i % 3], x=data[predictors[i]], y=reg_plot[i].predicted,
                        color='r', scatter=False, label='Predicted Y')
    else:
        for i in range(len(predictors)):
            sns.scatterplot(ax=axes_[i // 3, i % 3], x=data[predictors[i]], y=data['Y'],
                            alpha=0.5, color=point_colors[i])


def split_by_threshold(ss, predictor, threshold):
    return ss[ss[predictor] < threshold], ss[ss[predictor] >= threshold]


def get_decision_sets(source_set, conditions):
    return_set = []
    sample_set = source_set.copy()
    for condition in conditions:
        if condition == conditions[-1]:
            return_set.append(sample_set)
            break
        predictor, operator, threshold = condition[:2], condition[2], int(condition[3:])
        lt, gt = split_by_threshold(sample_set, predictor, threshold)
        if operator == '<':
            return_set.append(lt)
            sample_set = gt
        else:
            return_set.append(gt)
            sample_set = lt
    return return_set


def get_decision_models(source_set, conditions, return_sets=False):
    decisions = []
    conditional_sets = get_decision_sets(source_set, [condition['condition'] for condition in conditions])
    for cond_set, cond in zip(conditional_sets, conditions):
        regression_model = cond['model'](x=cond_set[cond['predictors']],
                                         y=cond_set['Y'],
                                         inputs=cond['predictors'], arg=cond['arg'],
                                         keep_estimator=True)
        decisions.append(regression_model)
    if return_sets:
        return decisions, conditional_sets
    return decisions


# Regression Functions
def single_linear_regression(x, y, arg=None, inputs=None, keep_estimator=False):
    estimator = None
    cv_scores = cross_val_score(estimator=LinearRegression(),
                                X=x,
                                y=y,
                                cv=5,
                                scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=x,
                                       y=y,
                                       cv=5)
    if keep_estimator:
        estimator = LinearRegression()
        estimator.fit(x, y)
    return ModelResult({'model': 'single_linear_regression',
                        'arg': '',
                        'predictors': inputs,
                        'predicted': cv_predictions,
                        'error': cv_scores.sum() / len(cv_scores),
                        'estimator': estimator})


def multiple_linear_regression(x, y, arg=None, inputs=None, keep_estimator=False):
    estimator = LinearRegression()
    cv_scores = cross_val_score(estimator=estimator,
                                X=x[inputs],
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=x[inputs],
                                       y=y,
                                       cv=5)
    if keep_estimator:
        estimator.fit(x, y)
    return ModelResult({'model': 'multiple_linear_regression',
                        'arg': '',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum()/len(cv_scores),
                        'estimator': estimator})


def polynomial_regression(x, y, arg=None, inputs=None, keep_estimator=False):
    poly_features = PolynomialFeatures(degree=arg, include_bias=False).fit_transform(x[inputs])
    estimator = LinearRegression()
    cv_scores = cross_val_score(estimator=estimator,
                                X=poly_features,
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=estimator,
                                       X=poly_features,
                                       y=y,
                                       cv=5)
    if keep_estimator:
        estimator.fit(poly_features, y)
    return ModelResult({'model': 'polynomial_regression',
                        'arg': f'order={arg}',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum()/len(cv_scores),
                        'estimator': estimator})


def k_nearest_regression(x, y, arg=5, inputs=None):
    knn_regressor = KNeighborsRegressor(n_neighbors=arg)
    cv_scores = cross_val_score(estimator=knn_regressor,
                                X=x,
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=knn_regressor,
                                       X=x,
                                       y=y,
                                       cv=5)
    return ModelResult({'model': 'k_nearest_regression',
                        'arg': f'k={arg}',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum() / len(cv_scores)})


def decision_tree_regression(x, y, arg=5, inputs=None):
    tree_regressor = DecisionTreeRegressor(random_state=0, max_depth=arg)
    cv_scores = cross_val_score(estimator=tree_regressor,
                                X=x[inputs],
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=tree_regressor,
                                       X=x[inputs],
                                       y=y,
                                       cv=5)
    return ModelResult({'model': 'decision_tree_regression',
                        'arg': f'max_depth={arg}',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum()/len(cv_scores)})


def random_forest_regression(x, y, arg=None, inputs=None):
    tree_regressor = RandomForestRegressor(max_depth=arg, random_state=0, max_features=len(inputs))
    cv_scores = cross_val_score(estimator=tree_regressor,
                                X=x[inputs],
                                y=y,
                                cv=5, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=tree_regressor,
                                       X=x[inputs],
                                       y=y,
                                       cv=5)
    return ModelResult({'model': 'random_forest_regression',
                        'arg': f'max_depth={arg}',
                        'predictors': ' '.join(inputs),
                        'predicted': cv_predictions,
                        'error': cv_scores.sum()/len(cv_scores)})