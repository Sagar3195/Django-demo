## Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

## Loading datasets
data = pd.read_csv("mpg_data_example.csv")

print(data.head())
print(data.shape)

##Let's check missing values in dataset
print(data.isnull().sum())

##We can see that there is 6 null values in horsepower feature
print(sns.distplot(data['horsepower']))
#plt.show()
print(data.info())

print(data.describe())

##Let's fill nan values
data['horsepower'] = data['horsepower'].fillna(data['horsepower'].mean())
##Let's check again null values in dataset
print(data.isnull().sum())

##We can see that there is no missing values in dataset

##Scaling dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

##Splitting dataset into independent variable and dependent variable
X = data.iloc[:,1:8]
y = data.iloc[:, 0]

print(X.head())
print(y.head())

##Now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state= 0)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#print(x_train[:5])
#print("----------------")
#print(x_test[:5])

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()

##Let's train the model
regressor.fit(x_train, y_train)

##Let's predict the model

predict = regressor.predict(x_test)
print(predict[:5])

#Let's check accuracy of the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
mse = mean_squared_error(y_test, predict)
print("Mean Squared Error: ", mse)

mae = mean_absolute_error(y_test, predict)
print("Mean Absolute Error: ", mae)

rmse = np.sqrt(mean_squared_error(y_test, predict))
print("Root mean squared error: ", rmse)

import joblib
#joblib.dump(regressor, 'mpg_model.pkl')

model_reload = joblib.load('mpg_model.pkl')

temp = {}
temp['cylinders'] = 1
temp["displacement"] = 2
temp["horsepower"] = 3
temp["weight"] = 4
temp["acceleration"] = 5
temp['model year'] = 6
temp['origin'] = 1

test_data = pd.DataFrame({'x':temp}).T
print(test_data)

print(model_reload.predict(test_data)[0])
