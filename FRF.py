import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV  # cross_validation is deprecated
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


def test_tree_bags(learning_groups, learning_data, testing_groups, testing_data, rf_type='classification', n_trees=10,
                   n_predictors="auto", oob_score=False, class_weight=None, feature_importance=False):
    """
    This takes at minimum 4 inputs, your learning group and data as well as testing group and data.

    rf_type can be chosen to be either classification or regression
    Read RandomForestClassifier documentation for more info on n_trees(n_estimators), n_predictors(max_features),
    oob_score, and class_weight.
    If a dictionary of feature importance is required set feature_importance to True
    
    This function will build the appropriate Random Forest based on parameters provided.
    It will fit the RF on your learning set.
    
    This function returns: 
    1) Predicted classes which are made using your testing data
    2) Overall accuracy of the RF in predicting your testing groups
    3) Predicted scores which are computed using the testing data
    4) Group accuracies for every group in your testing group
    
    Optionally, the function returns:
    5) Score of the training data set obtained using the out-of-bag estimate
    6) A dictionary of the features and how important they are to the RF model 
    """
    if rf_type == 'regression':
        tree_bag = RandomForestRegressor(n_estimators=n_trees, max_features=n_predictors, oob_score=oob_score)
    else:
        tree_bag = RandomForestClassifier(n_estimators=n_trees, max_features=n_predictors, oob_score=oob_score,
                                          class_weight=class_weight)

    tree_bag.fit(learning_data, learning_groups)
    predicted_classes = tree_bag.predict(testing_data)
    overall_accuracy = accuracy_score(testing_groups, predicted_classes)
    predicted_scores = tree_bag.predict_log_proba(testing_data)

    return_list = []

    if oob_score:
        oob_pred = tree_bag.oob_score_
        return_list.append(oob_pred)

    if feature_importance:
        features_dict = {}
        for i in range(len(tree_bag.feature_importances_)):
            features_dict[learning_data[i]] = tree_bag.feature_importances_[i]
            # check that it is getting the correct headers from learning set
        return_list.append(features_dict)

    return predicted_classes, predicted_scores, overall_accuracy, *return_list
