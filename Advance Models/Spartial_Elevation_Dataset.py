# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:54:04 2017

@author: sujay
"""
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# =============================================================================#
# 3D Spatial Dataset
# =============================================================================#

spatial = pd.read_csv('elevation.csv')

X = spatial.iloc[:,0:2]
y = spatial.iloc[:,3]    

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#----------------------------------------------------------------------------------#
# Split & Fit 
#----------------------------------------------------------------------------------#

# Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
    
# Predicting the Test set results
y_pred = regressor.predict(X_test)
    
#Evaluating the Model with r2_score Regression Metric
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Evaluating the model with RMSE metric
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test, y_pred))
   

#Decision Tree Regression
from sklearn import tree
reg = tree.DecisionTreeRegressor()
reg = reg.fit(X_train, y_train)

#Predicting the Test set results
y_pred = reg.predict(X_test)

#Evaluating the Model with r2_score Regression Metric
from sklearn.metrics import r2_score
a = r2_score(y_test, y_pred)
print("R2 score for = ",a)
    
#Evaluating the model with RMSE metric
from sklearn.metrics import mean_squared_error
b = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean Sq Error for = ",b)


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(max_depth=5, random_state = 0)
Regressor.fit(X_train, y_train)

#Predicting the values
y_pred = Regressor.predict(X_test)

#Evaluating the Model with r2_score Regression Metric
from sklearn.metrics import r2_score
a = r2_score(y_test, y_pred)
print("R2 score for = ",a)
    
#Evaluating the model with RMSE metric
from sklearn.metrics import mean_squared_error
b = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean Sq Error for = ",b)

#--------------------------------------------------------------------------------------------------------------------
# 10 Fold Cross Validation
#--------------------------------------------------------------------------------------------------------------------

# Linear Regression
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
y_pred = cross_val_predict(Regressor, X, y, cv = 10)


#Evaluating the Model with r2_score Regression Metric
from sklearn.metrics import r2_score
r2_score(y, y_pred)

#Evaluating the model with RMSE metric
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y, y_pred))

#Decision Tree
from sklearn.cross_validation import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
Regressor = DecisionTreeRegressor()
y_pred = cross_val_predict(Regressor, X, y, cv = 10)

#Evaluating model with r2_score metrics
from sklearn.metrics import r2_score
r2_score(y, y_pred)

#Evaluating the model with RMSE metric
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y, y_pred))


#Random Forest Regression
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor(max_depth=5, random_state = 0)
y_pred = cross_val_predict(Regressor, X, y, cv = 10)


#Evaluating model with r2_score metrics
from sklearn.metrics import r2_score
r2_score(y, y_pred)

#Evaluating the model with RMSE metric
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y, y_pred))


