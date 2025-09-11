
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
val = pd.read_csv('val.csv')


columns_dropped = ['WAIT_TIME_IN_2H',"DATETIME","dayofweek","month","minute","ENTITY_DESCRIPTION_SHORT"]
target = 'WAIT_TIME_IN_2H'

# X_train = data.drop(columns=columns_dropped)
y_train = data[target]

X_train = data.drop(columns=[c for c in columns_dropped if c in data.columns])
X_val   = val.drop(columns=[c for c in columns_dropped if c in val.columns])




from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05,n_jobs=-1,random_state=42)


xgb_model.fit(X_train, y_train, 
              
             verbose=False)

y_pred = xgb_model.predict(X_val)





#CSV de validation 

submission = pd.DataFrame({
    'DATETIME': val['DATETIME'],
    'ENTITY_DESCRIPTION_SHORT': val['ENTITY_DESCRIPTION_SHORT'],
    'y_pred': y_pred,
    'KEY': 'Validation'
})

submission.to_csv('submission_validation.csv', index=False)



#Test de RMSE sur le train split 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_t, X_v, y_t, y_v = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

xgb_model1 = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method="hist"  # plus rapide sur CPU
)

xgb_model1.fit(
    X_t, y_t,
    
    verbose=False
)

y_pred1 = xgb_model1.predict(X_v)

rmse = mean_squared_error(y_v, y_pred1, squared=False)
print(rmse)



### indicateurs de corrélation entre les paramètres     

import matplotlib.pyplot as plt
from xgboost import plot_importance

# Après fit :
# plot_importance(xgb_model, max_num_features=20)
# plt.show()

# print(list(zip(X_train.columns, xgb_model.feature_importances_)))

#donne un classement des variables les plus utiles pour le modèle.









import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# --- 0) Préparer les données de base (déjà dans ton script)
# X_train, y_train, xgb_model existent déjà ici

# --- 1) Choisir les colonnes numériques conservées
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# (Optionnel) limiter aux K colonnes les plus importantes selon XGBoost
#K = 10
#importances = dict(zip(X_train.columns, xgb_model.feature_importances_))
#num_cols = [c for c in sorted(num_cols, key=lambda c: importances.get(c, 0), reverse=True)[:K]]

# --- 2) Préparer le dossier de sortie
out_dir = "figs_pairplots"
os.makedirs(out_dir, exist_ok=True)

# --- 3) Normaliser la cible pour une intensité (0..1)
y = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
y_min, y_max = np.nanmin(y), np.nanmax(y)
y_norm = (y - y_min) / (y_max - y_min + 1e-9)  # évite division par 0

# --- 4) Boucle sur tous les couples de colonnes
pairs = list(combinations(num_cols, 2))
print(f"Nombre de paires à tracer : {len(pairs)}")

for i, (cx, cy) in enumerate(pairs, 1):
    x = X_train[cx].to_numpy()
    z = X_train[cy].to_numpy()

    plt.figure(figsize=(6, 5))
    # Nuage de points, intensité = y_norm (plus c'est élevé, plus c'est foncé)
    # cmap 'gray' : 0 clair -> 1 foncé ; s=7 pour lisibilité ; alpha léger pour réduire le sur-plotting
    sc = plt.scatter(x, z, c=y_norm, cmap='gray', s=7, alpha=0.6, edgecolors='none')
    plt.xlabel(cx)
    plt.ylabel(cy)
    plt.title(f"{cy} vs {cx} (intensité ~ WAIT_TIME_IN_2H)")
    cbar = plt.colorbar(sc)
    cbar.set_label("WAIT_TIME_IN_2H (normalisé)")

    plt.tight_layout()
    fname = os.path.join(out_dir, f"pair_{i:04d}_{cx}__vs__{cy}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

print(f"✅ Figures enregistrées dans: {out_dir}")
