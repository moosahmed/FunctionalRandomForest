import numpy as np
import pandas as pd
import timeit

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold  # cross_validation is deprecated
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

column_names = ['class_name', 'left_weight', 'left_distance', 'right_weight', 'right_distance']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                 header=None, names=column_names)

# print(df.head())
# print(df.info())
# print(df['class_name'].value_counts())

data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance']
target_cols = ['class_name']

X = df[data_cols]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # what is random state?
# rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=int(timeit.default_timer()))
# for train_index, test_index in rskf.split(X, y):
#     print("X:", X)
#     print("TRAIN:", train_index)
#     print("TEST:", test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

model = RandomForestClassifier()
model.fit(X_train, np.ravel(y_train))
# used np.ravel to change the y_train column vector to 1d array. .fit() expects a 1d array

y_predict = model.predict(X_test)
print(accuracy_score(y_test.values, y_predict))

# df['left_cross'] = df['left_distance']*df['left_weight']
# df['right_cross'] = df['right_distance']*df['right_weight']
# df['left_right_ratio'] = df['left_cross']/df['right_cross']
#
# new_data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance', 'left_cross', 'right_cross',
#                  'left_right_ratio']
# new_target_cols = ['class_name']
#
# Xx = df[new_data_cols]
# yy = df[new_target_cols]
# Xx_train, Xx_test, yy_train, yy_test = train_test_split(Xx, yy, test_size=0.3, random_state=42)
#
# forest = RandomForestClassifier().fit(Xx_train, np.ravel(yy_train))
# yy_predict = forest.predict(Xx_test)
# print(accuracy_score(yy_predict, yy_test))
#
#
# # To check how each feature contributes to a model
# features_dict = {}
# for i in range(len(forest.feature_importances_)):
#     features_dict[new_data_cols[i]] = forest.feature_importances_[i]
# print(sorted(features_dict.items(), key=lambda x: x[1], reverse=True))
#
# # hyperparameterization
# gridsearch_forest = RandomForestClassifier()
#
# params = {
#     "n_estimators": [100, 300, 500],
#     "max_depth": [5, 8, 15],
#     "min_samples_leaf": [1, 2, 4]
# }
#
# clf = GridSearchCV(gridsearch_forest, param_grid=params, cv=5).fit(Xx, np.ravel(yy))
# print(clf.best_params_)
# print(clf.best_score_)
