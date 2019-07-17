# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:28:36 2019

@author: Heisenberg
"""

import pandas as pd
import numpy as np

test=pd.read_excel('Data_Test.xlsx')

train=pd.read_excel('Data_Train.xlsx')
train['Location'].unique()
train['Year'].unique()
train['Fuel_Type'].unique()
train['Transmission'].unique()
train['Owner_Type'].unique()
train['Seats'].unique()

test['Location'].unique()
test['Year'].unique()
test['Fuel_Type'].unique()
test['Transmission'].unique()
test['Owner_Type'].unique()
test['Seats'].unique()

del train['Name']
del test['Name']
del train['Year']
del test['Year']
del train['New_Price']
del test['New_Price']

train['Mileage']=train['Mileage'].str.extract('(\d*.\d+)')
train['Engine']=train['Engine'].str.extract('(\d*.\d+)')
train['Power']=train['Power'].str.extract('(\d*.\d+)')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
train['Location'] = labelencoder_x.fit_transform(train['Location'])
train['Fuel_Type'] = labelencoder_x.fit_transform(train['Fuel_Type'])
train['Transmission'] = labelencoder_x.fit_transform(train['Transmission'])
train['Owner_Type'] = labelencoder_x.fit_transform(train['Owner_Type'])
dummy1=pd.get_dummies(train['Location'],prefix='Location')
dummy2=pd.get_dummies(train['Fuel_Type'],prefix='Fuel_Type')
dummy3=pd.get_dummies(train['Transmission'],prefix='Transmission')
dummy4=pd.get_dummies(train['Owner_Type'],prefix='Owner_Type')

train = pd.concat([train,dummy1],axis=1)
train = pd.concat([train,dummy2],axis=1)
train = pd.concat([train,dummy3],axis=1)
train = pd.concat([train,dummy4],axis=1)
del train['Location']
del train['Fuel_Type']
del train['Transmission']
del train['Owner_Type']

train['Mileage'].fillna(np.mean(train['Mileage'].astype(float)),inplace = True)
train['Engine'].fillna(int(np.mean(train['Engine'].astype(float))),inplace = True)
train['Power'].fillna(np.mean(train['Power'].astype(float)),inplace = True)
train['Seats'].fillna('5',inplace = True)

test['Mileage']=test['Mileage'].str.extract('(\d*.\d+)')
test['Engine']=test['Engine'].str.extract('(\d*.\d+)')
test['Power']=test['Power'].str.extract('(\d*.\d+)')

test['Location'] = labelencoder_x.fit_transform(test['Location'])
test['Fuel_Type'] = labelencoder_x.fit_transform(test['Fuel_Type'])
test['Transmission'] = labelencoder_x.fit_transform(test['Transmission'])
test['Owner_Type'] = labelencoder_x.fit_transform(test['Owner_Type'])

test_dummy1=pd.get_dummies(test['Location'],prefix='Location')
test_dummy2=pd.get_dummies(test['Fuel_Type'],prefix='Fuel_Type')
test_dummy3=pd.get_dummies(test['Transmission'],prefix='Transmission')
test_dummy4=pd.get_dummies(test['Owner_Type'],prefix='Owner_Type')

test = pd.concat([test,test_dummy1],axis=1)
test = pd.concat([test,test_dummy2],axis=1)
test = pd.concat([test,test_dummy3],axis=1)
test = pd.concat([test,test_dummy4],axis=1)
del test['Location']
del test['Fuel_Type']
del test['Transmission']
del test['Owner_Type']
test['Engine'].fillna(int(np.mean(test['Engine'].astype(float))),inplace = True)
test['Power'].fillna(np.mean(test['Power'].astype(float)),inplace = True)
test['Seats'].fillna('5',inplace = True)

test.apply(lambda x: sum(x.isnull()))
train.apply(lambda x: sum(x.isnull()))

del train['Fuel_Type_4']
import statistics
train['Mileage'] = pd.to_numeric(train['Mileage'], downcast='signed',errors='coerce')
train['Engine'] = pd.to_numeric(train['Engine'], downcast='float',errors='coerce')
train['Power'] = pd.to_numeric(train['Power'], downcast='signed',errors='coerce')
train['Seats'] = pd.to_numeric(train['Seats'], downcast='signed',errors='coerce')

test['Mileage'] = pd.to_numeric(test['Mileage'], downcast='signed',errors='coerce')
test['Engine'] = pd.to_numeric(test['Engine'], downcast='float',errors='coerce')
test['Power'] = pd.to_numeric(test['Power'], downcast='signed',errors='coerce')
test['Seats'] = pd.to_numeric(test['Seats'], downcast='signed',errors='coerce')
#scaling
"""train['Kilometers_Driven']=abs((train['Kilometers_Driven']-np.mean(train['Kilometers_Driven']))/statistics.stdev(train['Kilometers_Driven']))
description=train.describe()
q = train["Kilometers_Driven"].quantile(0.99)
train=train[train["Kilometers_Driven"] < q]
r = train["Price"].quantile(0.99)
train=train[train["Price"] < r]
train["Mileage"]=train["Mileage"]/33.54
train["Engine"]=(train["Engine"]-72)/(5461-72)
train['Power']=abs((train['Power']-np.mean(train['Power']))/statistics.stdev(train['Power']))
train["Seats"]=(train["Seats"])/(10)


test['Kilometers_Driven']=abs((test['Kilometers_Driven']-np.mean(test['Kilometers_Driven']))/statistics.stdev(test['Kilometers_Driven']))
description2=test.describe()
s = test["Kilometers_Driven"].quantile(0.99)
test=test[train["Kilometers_Driven"] < s]

test["Mileage"]=test["Mileage"]/32.26
test["Engine"]=(test["Engine"]-624)/(5998-624)
test['Power']=abs((test['Power']-np.mean(test['Power']))/statistics.stdev(test['Power']))
test["Seats"]=(test["Seats"]-2)/(10-2)
"""
#model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 512, random_state = 0)
regressor.fit(train.iloc[:,train.columns != 'Price'], train.iloc[:,5])



# Predicting the Test set results
y_pred = regressor.predict(test)
y_pred=pd.DataFrame(y_pred)
mm=train.describe()
submission=y_pred.to_excel("submission.xlsx")
  
