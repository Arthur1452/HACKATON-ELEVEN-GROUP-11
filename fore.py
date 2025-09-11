import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")


data['DATETIME'] = pd.to_datetime(data['DATETIME'])
data['hour'] = data['DATETIME'].dt.hour
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['month'] = data['DATETIME'].dt.month

data = data.drop(columns=['TIME_TO_PARADE_2']).copy()
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])

X = data.drop(columns=['DATETIME', 'WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']


# Supprimer la colonne inutilisable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Remplir les NaN par 0 dans les colonnes parade/show
for col in ['TIME_TO_PARADE_1', 'TIME_TO_NIGHT_SHOW']:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("RMSE interne (RandomForest):", rmse_rf)

# =======================
# 7️⃣ LinearRegression
# =======================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("RMSE interne (LinearRegression):", rmse_lr)

# =======================
# 8️⃣ Visualisation des prédictions
# =======================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.3)
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("RandomForest : prédictions vs réel")

plt.subplot(1,2,2)
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.3, color='orange')
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("LinearRegression : prédictions vs réel")
plt.tight_layout()
plt.show()