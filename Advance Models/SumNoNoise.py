#%%
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_predict
import configparser
import zipfile
import warnings
warnings.filterwarnings('ignore')

#%%
parser = configparser.ConfigParser()
parser.read('config.ini')
D1= parser.get('FILE_PATH','D1_P')

TEMP = parser.get('FILE_PATH','TEMP')

zippy = zipfile.ZipFile(D1,"r")
zippy.extractall(TEMP)



#%% Decision Tree regression

#Reading the excel into our Dataset
dataSet = pd.read_csv('The SUM dataset, without noise', delimiter= ';')

#Splitting the dataset X & y variables
X = dataSet.iloc[:,0:10].values
y = dataSet.iloc[:,11:12].values

#Creating a Model for our prediction Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
Regressor = DecisionTreeRegressor()
Regressor.fit(X_train, y_train)

#Predicting the values
y_pred = Regressor.predict(X_test)

#Evaluating model with r2_score metrics
r2_score(y, y_pred)

#Evaluating model with mean square error
mean_squared_error(y, y_pred)

#10 fold valuation
y_pred = cross_val_predict(Regressor, X, y, cv = 10)


#%% Simple linear regression
# Importing the dataset
dataset = pd.read_csv('The SUM dataset, without noise',delimiter=';')

#Understanding Correlation between columns
corrMatrix = dataset.corr()
#print corrMatrix
print(corrMatrix["Target"].sort_values(ascending=False))

#Creating X and y variables
X = dataset.iloc[: , 0:10 ].values
y = dataset.iloc[: , 11:12].values
     

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
                                                    random_state = 0)

#Training our Model with Linear Regression
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

#Predecting outsome using our Model
y_pred = Regressor.predict(X_test)

#Evaluating the Model with r2_score Regression Metric
r2_score(y_test, y_pred)

#Evaluating the model with RMSE metric
np.sqrt(mean_squared_error(y_test, y_pred))

#10 fold validation
y_pred = cross_val_predict(Regressor, X, y, cv = 10)

#%% random forest
#Reading the excel into our Dataset
dataSet = pd.read_csv('The SUM dataset, without noise', delimiter= ';')

#Splitting the dataset X & y variables
X = dataSet.iloc[:,0:10].values
y = dataSet.iloc[:,11].values

#Creating a Model for our prediction Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


Regressor = RandomForestRegressor(max_depth=5, random_state = 0)
Regressor.fit(X_train, y_train)

#Predicting the values
y_pred = Regressor.predict(X_test)

#Evaluating model with r2_score metrics
r2_score(y_test, y_pred)

#Evaluating model with r2_score metrics
mean_squared_error(y_test, y_pred)

#10 fold validation
y_pred = cross_val_predict(Regressor, X, y, cv = 10)
#%%
