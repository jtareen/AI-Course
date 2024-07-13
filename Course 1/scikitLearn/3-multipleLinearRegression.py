from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# get the absolute path of data
script_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_path, 'data/Student_Performance.csv')

# read the data using pandas
data = pd.read_csv(csv_path)

# Extracurricular Activities feature has text data with two categories -> map it to numeric data
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# split the data into train and test sets using sklearn train_test_split function
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# sepcify the features
indpdnt_vars = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']

# specify the feature and label of train set
x_train = train_set[indpdnt_vars].to_numpy()
y_train = np.array(train_set['Performance Index'])

# specify the feature and label of test set
x_test = test_set[indpdnt_vars].to_numpy()
y_test = np.array(test_set['Performance Index'])

# scale function
def scale_data(x):
    # initialize the standard scaler
    scaler = StandardScaler()

    # transform the data using the scaler
    x_scaled = scaler.fit_transform(x)
    return x_scaled

# linear model train function
def train_linear_model(x, y):
    # make linear model
    model = LinearRegression()

    # fit model to the data
    model.fit(x, y)

    return model

# polynomial model train function
def train_polynomial_model(x_train, y_train, deg):
    # make a polynomial with degree 3
    model_polynomial_features = PolynomialFeatures(degree=deg)

    # make a polynomial model with a pipeline
    model = make_pipeline(model_polynomial_features, LinearRegression())

    # fit data to the model
    model.fit(x_train, y_train)

    return model

# scale the train data
x_scaled = scale_data(x_train)

# train models
model1 = train_linear_model(x_scaled, y_train)
model2 = train_polynomial_model(x_scaled, y_train, 6)

print('Linear model score',r2_score(y_train, model1.predict(x_train)))
print('polynomial model score',r2_score(y_train, model2.predict(x_train)))

print('Linear model score on test set',r2_score(y_test, model1.predict(x_test)))
print('polynomial model score',r2_score(y_test, model2.predict(x_test)))