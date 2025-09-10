# =======================
# Notebook : Prédiction temps d'attente Euro-park
# =======================

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# =======================
# 0️⃣ Charger les données
# =======================
data = pd.read_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")
data['DATETIME'] = pd.to_datetime(data['DATETIME'])

weather = pd.read_csv("C:/Users/ouzen/Downloads/weather_data.csv", sep=",")
weather['DATETIME'] = pd.to_datetime(weather['DATETIME'])

# =======================
# 1️⃣ Features temporelles
# =======================
data['hour'] = data['DATETIME'].dt.hour
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['month'] = data['DATETIME'].dt.month
data['is_weekend'] = data['dayofweek'].isin([5,6]).astype(int)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

bins = [0, 10, 14, 18, 24]
labels = ['early','midday','afternoon','evening']
data['hour_block'] = pd.cut(data['hour'], bins=bins, labels=labels, right=False)
data = pd.get_dummies(data, columns=['hour_block'])

data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])
data = data.drop(columns=['TIME_TO_PARADE_2']).copy()

# =======================
# 2️⃣ Lags et rolling features
# =======================
for col in [c for c in data.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]:
    mask = data[col]==1
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_1'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(1)
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_2'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(2)
    data.loc[mask, 'CURRENT_WAIT_TIME_roll_1h'] = data.loc[mask, 'CURRENT_WAIT_TIME'].rolling(window=4).mean()

lag_cols = [c for c in data.columns if 'lag' in c or 'roll' in c]
data[lag_cols] = data[lag_cols].fillna(0)

# =======================
# 3️⃣ Merge météo
# =======================
weather_features = ['temp','humidity','wind_speed','rain_1h','snow_1h']
weather[weather_features] = weather[weather_features].fillna(0)
data = data.merge(weather[['DATETIME'] + weather_features], on='DATETIME', how='left')

# =======================
# 4️⃣ Préparer X et y
# =======================
X = data.drop(columns=['DATETIME','WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']

for col in ['TIME_TO_PARADE_1','TIME_TO_NIGHT_SHOW']:
    if col in X.columns:
        X[col] = X[col].fillna(0)

# =======================
# 5️⃣ GridSearch XGBoost
# =======================
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators':[100,200],
    'max_depth':[5,7],
    'learning_rate':[0.05,0.1],
    'subsample':[0.8,1.0]
}

grid_search = GridSearchCV(xgb_model,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X,y)
print("Meilleurs paramètres XGBoost :", grid_search.best_params_)

# =======================
# 6️⃣ Réentraîner sur tout le train
# =======================
xgb_final_full = xgb.XGBRegressor(**grid_search.best_params_, objective='reg:squarederror', random_state=42)
xgb_final_full.fit(X, y)

# =======================
# 7️⃣ Préparer le test de validation
# =======================
test_val = pd.read_csv("C:/Users/ouzen/Downloads/waiting_times_X_test_val.csv", sep=",")
test_val['DATETIME'] = pd.to_datetime(test_val['DATETIME'])

# Conserver une copie de la colonne pour le CSV
test_val['ENTITY_DESCRIPTION_SHORT_orig'] = test_val['ENTITY_DESCRIPTION_SHORT']

# Features temporelles
test_val['hour'] = test_val['DATETIME'].dt.hour
test_val['dayofweek'] = test_val['DATETIME'].dt.dayofweek
test_val['month'] = test_val['DATETIME'].dt.month
test_val['is_weekend'] = test_val['dayofweek'].isin([5,6]).astype(int)
test_val['hour_sin'] = np.sin(2 * np.pi * test_val['hour'] / 24)
test_val['hour_cos'] = np.cos(2 * np.pi * test_val['hour'] / 24)
test_val['hour_block'] = pd.cut(test_val['hour'], bins=bins, labels=labels, right=False)
test_val = pd.get_dummies(test_val, columns=['hour_block'])
test_val = pd.get_dummies(test_val, columns=['ENTITY_DESCRIPTION_SHORT'])

for col in ['TIME_TO_PARADE_1','TIME_TO_NIGHT_SHOW']:
    if col in test_val.columns:
        test_val[col] = test_val[col].fillna(0)

# Merge météo
test_val = test_val.merge(weather[['DATETIME'] + weather_features], on='DATETIME', how='left')
for col in weather_features:
    test_val[col] = test_val[col].fillna(0)

# Ajouter colonnes manquantes pour correspondre au train
for col in X.columns:
    if col not in test_val.columns:
        test_val[col] = 0

# Préparer X_test_val pour la prédiction
X_test_val = test_val[X.columns]

# =======================
# 8️⃣ Prédiction
# =======================
y_val_pred = xgb_final_full.predict(X_test_val)

# =======================
# 9️⃣ CSV de soumission pour validation
# =======================
submission = pd.DataFrame({
    'DATETIME': test_val['DATETIME'],
    'ENTITY_DESCRIPTION_SHORT': test_val['ENTITY_DESCRIPTION_SHORT_orig'],
    'y_pred': y_val_pred,
    'KEY': 'Validation'
})

submission.to_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/submission_validation_meteo.csv", index=False)

print("CSV de validation avec météo généré !")
