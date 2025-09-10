# =======================
# Notebook : Prédiction temps d'attente Euro-park
# =======================


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# =======================
# 0️⃣ Charger les données
# =======================
train_path = "C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv"
weather_path = "C:/Users/ouzen/Downloads/weather_data.csv"
val_path = "C:/Users/ouzen/Downloads/waiting_times_X_test_val.csv"

data = pd.read_csv(train_path, sep=",")
data['DATETIME'] = pd.to_datetime(data['DATETIME'])

weather = pd.read_csv(weather_path, sep=",")
weather['DATETIME'] = pd.to_datetime(weather['DATETIME'])

# =======================
# 1️⃣ Preprocessing météo (train + val)
# =======================
def preprocess_weather(weather_df):
    weather_df = weather_df.copy()
    weather_df['rain_1h'] = weather_df['rain_1h'].fillna(0)
    weather_df['snow_1h'] = weather_df['snow_1h'].fillna(0)
    weather_df['rain_1h_roll'] = weather_df['rain_1h'].rolling(window=4, min_periods=1).mean()
    weather_df['snow_1h_roll'] = weather_df['snow_1h'].rolling(window=4, min_periods=1).mean()
    weather_df['temp_roll'] = weather_df['temp'].rolling(window=4, min_periods=1).mean()
    weather_df['rain_cat'] = pd.cut(weather_df['rain_1h_roll'], bins=[-0.1,0.1,2,5,100], labels=[0,1,2,3])
    weather_df['snow_cat'] = pd.cut(weather_df['snow_1h_roll'], bins=[-0.1,0.1,2,5,100], labels=[0,1,2,3])
    weather_df = pd.get_dummies(weather_df, columns=['rain_cat','snow_cat'])
    return weather_df

weather = preprocess_weather(weather)
weather_features = ['temp_roll'] + [c for c in weather.columns if 'rain_cat_' in c or 'snow_cat_' in c or 'snow_cat_' in c]

# Merge météo
data = data.merge(weather[['DATETIME'] + weather_features], on='DATETIME', how='left')
for col in weather_features:
    data[col] = data[col].fillna(0)

# =======================
# 2️⃣ Features temporelles
# =======================
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['DATETIME'].dt.hour
    df['dayofweek'] = df['DATETIME'].dt.dayofweek
    df['month'] = df['DATETIME'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    bins = [0,10,14,18,24]
    labels = ['early','midday','afternoon','evening']
    df['hour_block'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)
    df = pd.get_dummies(df, columns=['hour_block'])
    return df

data = add_time_features(data)
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])
data = data.drop(columns=['TIME_TO_PARADE_2'], errors='ignore')

# =======================
# 3️⃣ Lags et rolling features
# =======================
for col in [c for c in data.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]:
    mask = data[col]==1
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_1'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(1)
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_2'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(2)
    data.loc[mask, 'CURRENT_WAIT_TIME_roll_1h'] = data.loc[mask, 'CURRENT_WAIT_TIME'].rolling(window=4, min_periods=1).mean()

lag_cols = [c for c in data.columns if 'lag' in c or 'roll' in c]
data[lag_cols] = data[lag_cols].fillna(0)

# =======================
# 4️⃣ Définir X et y
# =======================
X = data.drop(columns=['DATETIME','WAIT_TIME_IN_2H'])
y = data['WAIT_TIME_IN_2H']

for col in ['TIME_TO_PARADE_1','TIME_TO_NIGHT_SHOW']:
    if col in X.columns:
        X[col] = X[col].fillna(0)

X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================
# 5️⃣ GridSearch XGBoost
# =======================
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators':[100,200],
    'max_depth':[3,5,7],
    'learning_rate':[0.05,0.1],
    'subsample':[0.8,1.0]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_full, y_train_full)
print("Meilleurs paramètres XGBoost :", grid_search.best_params_)

# =======================
# 6️⃣ Réentraîner sur tout le train
# =======================
xgb_final = xgb.XGBRegressor(**grid_search.best_params_, objective='reg:squarederror', random_state=42)
xgb_final.fit(X_train_full, y_train_full)

y_val_pred = xgb_final.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE sur validation :", rmse_val)

# =======================
# 7️⃣ Préparer le test de validation
# =======================
test_val = pd.read_csv(val_path)
test_val['DATETIME'] = pd.to_datetime(test_val['DATETIME'])
test_val['ENTITY_DESCRIPTION_SHORT_orig'] = test_val['ENTITY_DESCRIPTION_SHORT']

# Appliquer les mêmes transformations
test_val = add_time_features(test_val)
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

X_test_val = test_val[X.columns]

# =======================
# 8️⃣ Prédiction + CSV
# =======================
y_test_val_pred = xgb_final.predict(X_test_val)

submission = pd.DataFrame({
    'DATETIME': test_val['DATETIME'],
    'ENTITY_DESCRIPTION_SHORT': test_val['ENTITY_DESCRIPTION_SHORT_orig'],
    'y_pred': y_test_val_pred,
    'KEY': 'Validation'
})

submission.to_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/submission_meteo.csv", index=False)
print("CSV de validation harmonisé généré !")
