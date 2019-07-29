"""Create dataframe showing the feature importance or relevance"""
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

RANDOM_STATE = 0
random.seed(RANDOM_STATE)


def predict_feature(feature_to_predict: str, data: pd.core.frame.DataFrame):
    """Use other feautures to predict the value of other feature"""
    X = data.copy()
    y = X[feature_to_predict]
    X.drop([feature_to_predict], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    importances = regressor.feature_importances_
    prediction_score = regressor.score(X_test, y_test)
    cols = X.columns.to_list()
    dat = {x[0]: x[1] for x in zip(cols, importances) }
    dat[feature_to_predict] = 0
    return prediction_score, dat
    
    
def cc_feature_relevance(raw_data: pd.core.frame.DataFrame):
    data = raw_data.copy()
    data.drop(['CUST_ID'], axis=1, inplace=True)
    data.dropna(inplace=True)
    features = ["_predicted_feature", "_score"]
    features.extend(data.columns.to_list())
    features_data = {feature: [] for feature in features}
    for feature in features[2:]:
        score, feature_importance = predict_feature(feature, data)
        features_data["_predicted_feature"].append(feature)
        features_data["_score"].append(score)
        for _featr, relevance in feature_importance.items():
            features_data[_featr].append(relevance)
        
    return pd.DataFrame(data=features_data)

