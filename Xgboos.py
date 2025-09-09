import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
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
# 1️⃣ Preprocessing météo
# =======================
weather['rain_1h'] = weather['rain_1h'].fillna(0)
weather['snow_1h'] = weather['snow_1h'].fillna(0)

# Rolling moyenne sur 1h (4*15min)
weather['rain_1h_roll'] = weather['rain_1h'].rolling(window=4).mean().fillna(0)
weather['snow_1h_roll'] = weather['snow_1h'].rolling(window=4).mean().fillna(0)
weather['temp_roll'] = weather['temp'].rolling(window=4).mean().fillna(0)

# Binning pluie et neige
weather['rain_cat'] = pd.cut(weather['rain_1h_roll'], bins=[-0.1,0.1,2,5,100], labels=[0,1,2,3])
weather['snow_cat'] = pd.cut(weather['snow_1h_roll'], bins=[-0.1,0.1,2,5,100], labels=[0,1,2,3])

# One-hot encoding catégories météo
weather = pd.get_dummies(weather, columns=['rain_cat','snow_cat'])

# Merge avec dataset principal
data = data.merge(weather[['DATETIME','temp_roll'] + [c for c in weather.columns if 'rain_cat_' in c or 'snow_cat_' in c]],
                  on='DATETIME', how='left')
for col in ['temp_roll'] + [c for c in weather.columns if 'rain_cat_' in c or 'snow_cat_' in c]:
    data[col] = data[col].fillna(0)

# =======================
# 2️⃣ Features temporelles
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

# Supprimer colonne inutilisable
data = data.drop(columns=['TIME_TO_PARADE_2']).copy()
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])

# =======================
# 3️⃣ Lags et rolling features sur CURRENT_WAIT_TIME
# =======================
for col in [c for c in data.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]:
    mask = data[col]==1
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_1'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(1)
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_2'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(2)
    data.loc[mask, 'CURRENT_WAIT_TIME_roll_1h'] = data.loc[mask, 'CURRENT_WAIT_TIME'].rolling(window=4).mean()

lag_cols = [c for c in data.columns if 'lag' in c or 'roll' in c]
data[lag_cols] = data[lag_cols].fillna(0)

# =======================
# 4️⃣ Définir X et y
# =======================
X = data.drop(columns=['DATETIME','WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']

X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Remplir NaN parade/show
for col in ['TIME_TO_PARADE_1','TIME_TO_NIGHT_SHOW']:
    if col in X_train_full.columns:
        X_train_full[col] = X_train_full[col].fillna(0)
        X_val[col] = X_val[col].fillna(0)

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

grid_search = GridSearchCV(xgb_model,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X_train_full,y_train_full)
print("Meilleurs paramètres XGBoost :", grid_search.best_params_)

# =======================
# 6️⃣ Modèle final XGBoost
# =======================
xgb_final = xgb.XGBRegressor(**grid_search.best_params_, objective='reg:squarederror', random_state=42)
xgb_final.fit(X_train_full,y_train_full)

y_val_pred_xgb = xgb_final.predict(X_val)
rmse_val_xgb = np.sqrt(mean_squared_error(y_val,y_val_pred_xgb))
print("RMSE XGBoost sur validation :", rmse_val_xgb)

# =======================
# 7️⃣ Visualisation
# =======================
plt.figure(figsize=(12,5))
sns.scatterplot(x=y_val, y=y_val_pred_xgb, alpha=0.3)
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("XGBoost : prédictions vs réel")
plt.show()
