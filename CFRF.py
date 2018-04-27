from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rftype = 'classification'
ntrees = 4
oob_score = True
numpredictors = 3
prior = 'balanced_subsample'


forest_parser = {}
forest_parser['n_estimators'] = ntrees
if numpredictors:
    forest_parser['max_features'] = numpredictors
forest_parser['oob_score'] = oob_score
if prior:
    forest_parser['class_weight'] = prior

if rftype == 'regression':
    print(RandomForestRegressor(**forest_parser))
else:
    print(RandomForestClassifier(**forest_parser))

