import numpy as np
import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")

data['DATETIME'] = pd.to_datetime(data['DATETIME'])

data['hour'] = data['DATETIME'].dt.hour
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['month'] = data['DATETIME'].dt.month

data = data.drop(columns=['TIME_TO_PARADE_2']).copy()

data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])


X = data.drop(columns=['DATETIME', 'WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']

data_val=pd.read_csv("C:/Users/ouzen/Downloads/waiting_times_X_test_val.csv", sep=",")
# 1️⃣ Supprimer TIME_TO_PARADE_2
data_val = data_val.drop(columns=['TIME_TO_PARADE_2'])

# 2️⃣ Remplir les NaN des autres colonnes par 0 (ou par la médiane si tu veux)
data_val['TIME_TO_PARADE_1'] = data_val['TIME_TO_PARADE_1'].fillna(0)
data_val['TIME_TO_NIGHT_SHOW'] = data_val['TIME_TO_NIGHT_SHOW'].fillna(0)

# 3️⃣ Extraire features temporelles
data_val['DATETIME'] = pd.to_datetime(data_val['DATETIME'])
data_val['hour'] = data_val['DATETIME'].dt.hour
data_val['dayofweek'] = data_val['DATETIME'].dt.dayofweek
data_val['month'] = data_val['DATETIME'].dt.month

# 4️⃣ One-hot encoder les parcs
data_val = pd.get_dummies(data_val, columns=['ENTITY_DESCRIPTION_SHORT'])

for col in X.columns:
    if col not in data_val.columns:
        data_val[col] = 0

X_val = data_val[X.columns]  # exactement les features du train

best_params = {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
rf_final = RandomForestRegressor(**best_params, random_state=42)
rf_final.fit(X, y)

y_val_pred = rf_final.predict(X_val)

output = pd.DataFrame({
    'DATETIME': data_val['DATETIME'],
    'ENTITY_DESCRIPTION_SHORT': pd.read_csv("C:/Users/ouzen/Downloads/waiting_times_X_test_val.csv")['ENTITY_DESCRIPTION_SHORT'],
    'WAIT_TIME_IN_2H': y_val_pred,
    'KEY': 'Validation'
})

