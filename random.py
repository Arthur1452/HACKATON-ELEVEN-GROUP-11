import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =======================
# 1️⃣ Charger et préparer le train set
# =======================
data = pd.read_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")

data['DATETIME'] = pd.to_datetime(data['DATETIME'])
data['hour'] = data['DATETIME'].dt.hour
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['month'] = data['DATETIME'].dt.month

data = data.drop(columns=['TIME_TO_PARADE_2']).copy()
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])

X = data.drop(columns=['DATETIME', 'WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']

# =======================
# 2️⃣ Split interne pour tester le modèle
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 3️⃣ Entraîner RandomForest
# =======================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =======================
# 4️⃣ Tester sur le split interne
# =======================
y_pred = model.predict(X_test)

# Évaluer la performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE interne:", mse)
print("R² interne:", r2)

# Visualiser
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel("Temps d'attente réel")
plt.ylabel("Temps d'attente prédit")
plt.title("RandomForest : prédictions sur split interne")
plt.show()

# =======================
# 5️⃣ Générer les prédictions sur le vrai fichier de validation
# =======================
data_val = pd.read_csv("C:/Users/ouzen/Downloads/waiting_times_X_test_val.csv", sep=",")

data_val = data_val.drop(columns=['TIME_TO_PARADE_2'])
data_val['TIME_TO_PARADE_1'] = data_val['TIME_TO_PARADE_1'].fillna(0)
data_val['TIME_TO_NIGHT_SHOW'] = data_val['TIME_TO_NIGHT_SHOW'].fillna(0)

data_val['DATETIME'] = pd.to_datetime(data_val['DATETIME'])
data_val['hour'] = data_val['DATETIME'].dt.hour
data_val['dayofweek'] = data_val['DATETIME'].dt.dayofweek
data_val['month'] = data_val['DATETIME'].dt.month

data_val = pd.get_dummies(data_val, columns=['ENTITY_DESCRIPTION_SHORT'])

# Aligner les colonnes
for col in X.columns:
    if col not in data_val.columns:
        data_val[col] = 0

X_val = data_val[X.columns]

# Prédictions finales
y_val_pred = model.predict(X_val)
data_val['PREDICTED_WAIT_TIME'] = y_val_pred