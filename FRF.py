import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV  # cross_validation is deprecated
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


def testTreeBags(learning_groups, learning_data, testing_groups, testing_data, ntrees, ttype, Qtreebag, treeweights,
                 categorical_vector, numpredictors=0):
    '''
    X,Y are determined in .fit()
    number of predictors(max_features) is by default set to: max_features=sqrt(n_features) only if unspecified by the user. if specified use that int

    dont know for sure how to include categorical vector

    surrogate? is it the same as criterion or sample weight in fit(). not sure.

    prior?
    '''
    treebag = RandomForestClassifier(n_estimators=ntrees, max_features=numpredictors, class_weight=prior)
    # treebag = RandomForestRegressor(ntrees)
    treebag.fit(learning_data, learning_groups)
    predicted_groups = treebag.predict(testing_data)
    # predicted_scores ?
    accuracy = accuracy_score(testing_groups, predicted_groups)


