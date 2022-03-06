import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR


#Import data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data = data.drop(columns=['id'])

#Data droping NaN values
data_discard = data.dropna()

#Data replacing NaN values with column mean
data_mean = data.fillna(data.mean().round(1))

#Data replacing NaN values with Linear Regression
data_LR = data.copy()

#Data replacing NaN values with K-Neighbors Regression
data_KNN = data.copy()

#Data replacing NaN values with Bayesian Regression
data_Bayesian = data.copy()

#Data replacing NaN values with Stochastic Gradient Descent
data_SGD = data.copy()

#Data replacing NaN values with Support Vector Regression
data_SVR = data.copy()

def Regression(
    data: pd.DataFrame, pipeline: Pipeline
) -> pd.DataFrame:
    """Given a Pipeline performs data imputation on the DataFrame using Regression

    Parameters:
        data (pd.DataFrame): Selected DataFrame with missing values
        pipeline (Pipeline): Custom builed pipeline

    Prints:
        data: Prints DataFrame with replaced missing values

    Returns:
        data: Returning DataFrame with replaced missing values

   """
    x = data[['age','gender','bmi']].copy()
    x.gender = x.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
    Missing = x[x.bmi.isna()]
    x = x[~x.bmi.isna()]
    y = x.pop('bmi')
    pipeline.fit(x,y)
    predicted_bmi = pd.Series(pipeline.predict(Missing[['age','gender']]),index=Missing.index)
    data.loc[Missing.index,'bmi'] = predicted_bmi.round(1)
    print(data)
    return data

LR_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',LinearRegression())
                              ])

KNN_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('knn',KNeighborsRegressor())
                              ])

Bayesian_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('Bayesian',BayesianRidge())
                              ])

SGD_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('SGD',SGDRegressor())
                              ])

SVR_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('SGD',LinearSVR())
                              ])

print(data_discard)
print(data_mean)
data_LR = Regression(data_LR, LR_bmi_pipe)
data_KNN = Regression(data_KNN, KNN_bmi_pipe)
data_Bayesian = Regression(data_Bayesian, Bayesian_bmi_pipe)
data_SGD = Regression(data_SGD, SGD_bmi_pipe)
data_SVR = Regression(data_SVR, SVR_bmi_pipe)



def RandomForest(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Performs Random Forest Classification on given DataFrame

    Parameters:
        data (pd.DataFrame): Selected DataFrame
    
    Prints:
        data: Prints Classification's evaluation metrics

    Returns:
        predictions: Returning DataFrame with predicted values

   """    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    x = data.copy()
    y = x.pop('stroke')
    x = pd.get_dummies(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    clf.fit(X_train, y_train)
    predictions = pd.DataFrame(clf.predict(X_test))
    print(classification_report(y_test, predictions, zero_division=1))
    return predictions

RandomForest(data_discard)
RandomForest(data_mean)
RandomForest(data_LR)
RandomForest(data_KNN)
RandomForest(data_Bayesian)
RandomForest(data_SGD)
RandomForest(data_SVR)