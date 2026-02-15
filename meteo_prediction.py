import numpy as np
import pandas as pd

df_train = pd.read_csv('weather_prediction_data.csv')

#sélectionner les features et la target 
columns_dropped = ['rain_in_2h',"DATETIME"]
target = 'rain_in_2h'


### train model (XGBOOST)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


y_train = df_train[target]

X_train = df_train.drop(columns=[c for c in columns_dropped if c in df_train.columns])

X_t, X_v, y_t, y_v = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05,n_jobs=-1,random_state=42)


xgb_model.fit(X_t, y_t, 
              
             verbose=False)


### predict rain
y_pred = xgb_model.predict(X_v)

### check accuracy 

rmse = mean_squared_error(y_v, y_pred)
mae = mean_absolute_error(y_v, y_pred)
print(rmse)
print(mae)
#
0.038

### fill the holes in the weather data prediction 

