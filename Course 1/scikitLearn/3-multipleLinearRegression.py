from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

script_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_path, 'data/Student_Performance.csv')

data = pd.read_csv(csv_path)

data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

indpdnt_vars = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


x_train = train_set[indpdnt_vars].to_numpy()
y_train = np.array(train_set['Performance Index'])

x_test = test_set[indpdnt_vars].to_numpy()
y_test = np.array(test_set['Performance Index'])

def train_linear_model(x, y):
    model = LinearRegression()
    model.fit(x, y)

    return model

def train_polynomial_model(x_train, y_train, deg):
    # make a polynomial with degree 3
    model_polynomial_features = PolynomialFeatures(degree=deg)

    # make a polynomial model with a pipeline
    model = make_pipeline(model_polynomial_features, LinearRegression())

    # fit data to the model
    model.fit(x_train, y_train)

    return model

model1 = train_linear_model(x_train, y_train)
model2 = train_polynomial_model(x_train, y_train, 6)

print('Linear model score',r2_score(y_train, model1.predict(x_train)))
print('polynomial model score',r2_score(y_train, model2.predict(x_train)))


print('Linear model score on test set',r2_score(y_test, model1.predict(x_test)))
print('polynomial model score',r2_score(y_test, model2.predict(x_test)))