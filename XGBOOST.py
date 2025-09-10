
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

# X_val = val.drop(columns=columns_dropped)


from xgboost import XGBRegressor

# xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.025,n_jobs=-1,random_state=42)


# xgb_model.fit(X_train, y_train, 
              
#              verbose=False)

# y_pred = xgb_model.predict(X_val)


# param_grid = {
#     'n_estimators':[500],
#     'max_depth':[3,5,7,9],
#     'learning_rate':[0.05,0.1,0.15],
#     'subsample':[0.8,1.0]
# }

# grid_search = GridSearchCV(xgb_model,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
# grid_search.fit(X,y)
# print("Meilleurs paramètres XGBoost :", grid_search.best_params_)



# xgb_final_full = xgb.XGBRegressor(**grid_search.best_params_, objective='reg:squarederror', random_state=42)
# xgb_final_full.fit(X, y)



# Préparer X_test_val pour la prédiction







#CSV de validation 

# submission = pd.DataFrame({
#     'DATETIME': val['DATETIME'],
#     'ENTITY_DESCRIPTION_SHORT': val['ENTITY_DESCRIPTION_SHORT'],
#     'y_pred': y_pred,
#     'KEY': 'Validation'
# })

# submission.to_csv('submission_validation.csv', index=False)

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

y_pred = xgb_model1.predict(X_v)

rmse = mean_squared_error(y_v, y_pred, squared=False)
print(rmse)

y_pred = xgb_model1.predict(X_val)


#CSV de validation 

submission = pd.DataFrame({
    'DATETIME': val['DATETIME'],
    'ENTITY_DESCRIPTION_SHORT': val['ENTITY_DESCRIPTION_SHORT'],
    'y_pred': y_pred,
    'KEY': 'Validation'
})

submission.to_csv('submission_validation.csv', index=False)