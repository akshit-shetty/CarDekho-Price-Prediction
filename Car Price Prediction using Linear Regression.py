## Problem Statement:
# The CarDekho company maintains a database of the cars sold through their platform. The data represents the cars sold by Car Dekho and the car-related features. We have to build a linear regression model to predict the Selling price of the car. Calculate all the error metrics and diagnostic plots to check the regression result. The dataset contains data for around 301 cars sold by CarDekho.


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


# ## Read the cars dataset

car = pd.read_csv("car data.csv")

car.head(10)


## Exploratory Data Analysis (EDA) 

# Data quality check



print(car.info())
print(car.describe())


# Check for missing values

print(car.isnull().sum())


# Correlation matrix to identify linear relationships

correlation_matrix = car.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


## Transform the categorical data.
car_final = car.drop(['Car_Name','Selling_Price'], axis =1)

car_final = pd.get_dummies(car_final)
car_final.shape
car_final.head()


## Multiple Linear Regression model



X = car_final
X = stats.zscore(X)
X = sm.add_constant(X)
y = car['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

res = sm.OLS(y_train, X_train).fit()
print(res.summary())


## Feature Importance

feature_importance = pd.Series(res.params)
feature_importance.plot(kind='barh')
plt.title('Feature Importance (Statsmodels)')
plt.xlabel('Coefficients')
plt.show()


