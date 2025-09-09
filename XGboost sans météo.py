import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# =======================
# 1️⃣ Charger les données
# =======================
data = pd.read_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")
data['DATETIME'] = pd.to_datetime(data['DATETIME'])

# Features temporelles
data['hour'] = data['DATETIME'].dt.hour
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['month'] = data['DATETIME'].dt.month
data['is_weekend'] = data['dayofweek'].isin([5,6]).astype(int)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

# Tranches horaires
bins = [0, 10, 14, 18, 24]
labels = ['early', 'midday', 'afternoon', 'evening']
data['hour_block'] = pd.cut(data['hour'], bins=bins, labels=labels, right=False)
data = pd.get_dummies(data, columns=['hour_block'])

# Supprimer colonne inutilisable
data = data.drop(columns=['TIME_TO_PARADE_2']).copy()

# One-hot encoding des attractions
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])

# =======================
# 2️⃣ Lags et rolling features
# =======================
for col in [c for c in data.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]:
    mask = data[col] == 1
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_1'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(1)
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_2'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(2)
    data.loc[mask, 'CURRENT_WAIT_TIME_roll_1h'] = data.loc[mask, 'CURRENT_WAIT_TIME'].rolling(window=4).mean()

# Remplir NaN des lags par 0
lag_cols = [c for c in data.columns if 'lag' in c or 'roll' in c]
data[lag_cols] = data[lag_cols].fillna(0)

# =======================
# 3️⃣ Définir X et y
# =======================
X = data.drop(columns=['DATETIME', 'WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']

# Split train/validation
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 4️⃣ Remplir NaN parade/show
# =======================
for col in ['TIME_TO_PARADE_1', 'TIME_TO_NIGHT_SHOW']:
    if col in X_train_full.columns:
        X_train_full[col] = X_train_full[col].fillna(0)
        X_val[col] = X_val[col].fillna(0)

# =======================
# 5️⃣ GridSearch XGBoost
# =======================
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train_full, y_train_full)
print("Meilleurs paramètres XGBoost :", grid_search.best_params_)

# =======================
# 6️⃣ Modèle final XGBoost
# =======================
xgb_final = xgb.XGBRegressor(**grid_search.best_params_, objective='reg:squarederror', random_state=42)
xgb_final.fit(X_train_full, y_train_full)

y_val_pred_xgb = xgb_final.predict(X_val)
rmse_val_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
print("RMSE XGBoost sur validation :", rmse_val_xgb)

# =======================
# 7️⃣ Comparaison RandomForest
# =======================
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
rf_model.fit(X_train_full, y_train_full)
y_val_pred_rf = rf_model.predict(X_val)
rmse_val_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
print("RMSE RandomForest sur validation :", rmse_val_rf)

# =======================
# 8️⃣ Comparaison LinearRegression
# =======================
lr_model = LinearRegression()
lr_model.fit(X_train_full, y_train_full)
y_val_pred_lr = lr_model.predict(X_val)
rmse_val_lr = np.sqrt(mean_squared_error(y_val, y_val_pred_lr))
print("RMSE LinearRegression sur validation :", rmse_val_lr)

# =======================
# 9️⃣ Visualisation
# =======================
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.scatterplot(x=y_val, y=y_val_pred_xgb, alpha=0.3)
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("XGBoost : prédictions vs réel")

plt.subplot(1,3,2)
sns.scatterplot(x=y_val, y=y_val_pred_rf, alpha=0.3, color='green')
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("RandomForest : prédictions vs réel")

plt.subplot(1,3,3)
sns.scatterplot(x=y_val, y=y_val_pred_lr, alpha=0.3, color='orange')
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("LinearRegression : prédictions vs réel")

plt.tight_layout()
plt.show()
