import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =======================
# 1️⃣ Charger et préparer les données
# =======================
data = pd.read_csv("C:/Users/ouzen/Documents/HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")

data['DATETIME'] = pd.to_datetime(data['DATETIME'])
data['hour'] = data['DATETIME'].dt.hour
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['month'] = data['DATETIME'].dt.month

# Supprimer TIME_TO_PARADE_2
data = data.drop(columns=['TIME_TO_PARADE_2']).copy()
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])

X = data.drop(columns=['DATETIME', 'WAIT_TIME_IN_2H']).copy()
y = data['WAIT_TIME_IN_2H']

# Remplir les NaN par 0 dans les colonnes parade/show
for col in ['TIME_TO_PARADE_1', 'TIME_TO_NIGHT_SHOW']:
    if col in X.columns:
        X[col] = X[col].fillna(0)

# =======================
# 2️⃣ Cross-validation pour RandomForest
# =======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_model_cv = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model_cv, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-cv_scores)
print("RMSE CV par fold:", rmse_cv)
print("RMSE moyen CV:", rmse_cv.mean())

# =======================
# 3️⃣ GridSearch pour hyperparamètres
# =======================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X, y)
print("Meilleurs paramètres:", grid_search.best_params_)
best_rmse = np.sqrt(-grid_search.best_score_)
print("RMSE moyen CV avec meilleurs paramètres:", best_rmse)

# =======================
# 4️⃣ Split interne pour comparer RandomForest et LinearRegression
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("RMSE interne (RandomForest final):", rmse_rf)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("RMSE interne (LinearRegression):", rmse_lr)

# =======================
# 5️⃣ Visualisation
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
