import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from sklearn.model_selection import train_test_split


#Constantes qui serviront à remplir les lignes manquantes

const_lag15 =100
const_lag30 = 100
const_parade1 = 480
const_parade2 = 480
const_paradenight = 480
const_snow = 0

#lecture des data set
data = pd.read_csv("HACKATON-ELEVEN-GROUP-11/waiting_times_train.csv", sep=",")
meteo = pd.read_csv("HACKATON-ELEVEN-GROUP-11/weather_data.csv")

#merge avec le data set de météo
data = data.merge(meteo, on= "DATETIME", how = "left")
data['DATETIME'] = pd.to_datetime(data['DATETIME'])

weather = pd.read_csv("HACKATON-ELEVEN-GROUP-11/weather_data.csv", sep=",")
weather['DATETIME'] = pd.to_datetime(weather['DATETIME'])



#création des colonnes en lien avec le temps
data['minute'] = data['DATETIME'].dt.hour*60 + data['DATETIME'].dt.minute

data['month'] = data['DATETIME'].dt.month
data['dayofweek'] = data['DATETIME'].dt.dayofweek
data['hour']=data['DATETIME'].dt.hour
data["year"]=data['DATETIME'].dt.year

data['is_weekend'] = data['dayofweek'].isin([5]).astype(int)

data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 1440)
data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 1440)

# Encodage cyclique pour jour de semaine (cycle 7 jours)
data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)

# Encodage cyclique pour mois (cycle 12 mois)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)


# trie des attractions en binaires
data = pd.get_dummies(data, columns=['ENTITY_DESCRIPTION_SHORT'])



#même chose pour le validation set
val = pd.read_csv("HACKATON-ELEVEN-GROUP-11/waiting_times_X_test_val.csv", sep=",")
meteo = pd.read_csv("HACKATON-ELEVEN-GROUP-11/weather_data.csv")
val = val.merge(meteo, on= "DATETIME", how = "left")
val['DATETIME'] = pd.to_datetime(val['DATETIME'])

weather = pd.read_csv("HACKATON-ELEVEN-GROUP-11/weather_data.csv", sep=",")
weather['DATETIME'] = pd.to_datetime(weather['DATETIME'])

val['dayofweek'] = val['DATETIME'].dt.dayofweek

val['minute'] = val['DATETIME'].dt.hour*60 + val['DATETIME'].dt.minute
val['month'] = val['DATETIME'].dt.month
val['is_weekend'] = val['dayofweek'].isin([5]).astype(int)
val["year"]=val["DATETIME"].dt.year


val['minute_sin'] = np.sin(2 * np.pi * val['minute'] / 1440)
val['minute_cos'] = np.cos(2 * np.pi * val['minute'] / 1440)

# Encodage cyclique pour jour de semaine
val['dayofweek_sin'] = np.sin(2 * np.pi * val['dayofweek'] / 7)
val['dayofweek_cos'] = np.cos(2 * np.pi * val['dayofweek'] / 7)

# Encodage cyclique pour mois
val['month_sin'] = np.sin(2 * np.pi * val['month'] / 12)
val['month_cos'] = np.cos(2 * np.pi * val['month'] / 12)

# Conserver une copie de la colonne pour le CSV
val['hour']=val['DATETIME'].dt.hour
entity_desc = val["ENTITY_DESCRIPTION_SHORT"]

val = pd.get_dummies(val, columns=['ENTITY_DESCRIPTION_SHORT'])



#récupération de la pluie dans deux heures depuis le météo set
data['DATETIME'] = pd.to_datetime(data['DATETIME'])
meteo['DATETIME'] = pd.to_datetime(meteo['DATETIME'])
val['DATETIME']= pd.to_datetime(val['DATETIME'])

meteo_2h = meteo[["DATETIME","rain_1h"]]
data['datetime_plus_2h'] = data['DATETIME'] + pd.Timedelta(hours=2)
val['datetime_plus_2h'] = val['DATETIME'] + pd.Timedelta(hours=2)

data = data.merge(
    meteo_2h[['DATETIME', 'rain_1h']], 
    left_on='datetime_plus_2h', 
    right_on='DATETIME', 
    how='left',
    suffixes=('', '_meteo')
).drop(['DATETIME_meteo'], axis=1)

val = val.merge(    
    meteo_2h[['DATETIME', 'rain_1h']], 
    left_on='datetime_plus_2h', 
    right_on='DATETIME', 
    how='left',
    suffixes=('', '_meteo')
).drop(['DATETIME_meteo'], axis=1)



# récupération des temps d'attentes il y a 15 minutes et il y a 30 minutes
for col in [c for c in data.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]:
    mask = data[col]==1
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_1'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(1)
    data.loc[mask, 'CURRENT_WAIT_TIME_lag_2'] = data.loc[mask, 'CURRENT_WAIT_TIME'].shift(2)


lag_cols = [c for c in data.columns if 'lag' in c or 'roll' in c]

data['exists_minus_15'] = (~data['CURRENT_WAIT_TIME_lag_1'].isna()).astype(int)
data['exists_minus_30'] = (~data['CURRENT_WAIT_TIME_lag_2'].isna()).astype(int)



for col in [c for c in data.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]:
    mask = val[col]==1
    val.loc[mask, 'CURRENT_WAIT_TIME_lag_1'] = val.loc[mask, 'CURRENT_WAIT_TIME'].shift(1)
    val.loc[mask, 'CURRENT_WAIT_TIME_lag_2'] = val.loc[mask, 'CURRENT_WAIT_TIME'].shift(2)


lag_cols = [c for c in val.columns if 'lag' in c or 'roll' in c]

val['exists_minus_15'] = (~val['CURRENT_WAIT_TIME_lag_1'].isna()).astype(int)
val['exists_minus_30'] = (~val['CURRENT_WAIT_TIME_lag_2'].isna()).astype(int)



#remplissage des colonnes avec du nan
cols_check = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW', 'snow']

for col in cols_check:
    if col in data.columns:  # on vérifie que la colonne existe
        data[f'{col}_isna'] = data[col].isna().astype(int)

cols_check = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW', 'snow']

for col in cols_check:
    if col in data.columns:  # on vérifie que la colonne existe
        val[f'{col}_isna'] = val[col].isna().astype(int)

# création d'une colonne spéciale pour détecter les jours fériés
jours_feries = [
    "2018-01-01","2018-04-02","2018-05-01","2018-05-08","2018-05-10","2018-05-21","2018-07-14","2018-08-15","2018-11-01","2018-11-11","2018-12-25",
    "2019-01-01","2019-04-22","2019-05-01","2019-05-08","2019-05-30","2019-06-10","2019-07-14","2019-08-15","2019-11-01","2019-11-11","2019-12-25",
    "2020-01-01","2020-04-13","2020-05-01","2020-05-08","2020-05-21","2020-06-01","2020-07-14","2020-08-15","2020-11-01","2020-11-11","2020-12-25",
    "2021-01-01","2021-04-05","2021-05-01","2021-05-08","2021-05-13","2021-05-24","2021-07-14","2021-08-15","2021-11-01","2021-11-11","2021-12-25",
    "2022-01-01","2022-04-18","2022-05-01","2022-05-08","2022-05-26","2022-06-06","2022-07-14","2022-08-15","2022-11-01","2022-11-11","2022-12-25"
]

# Convertir en datetime.date
jours_feries = pd.to_datetime(jours_feries).date


# Créer la colonne is_holiday pour détecter les journées de vacances scolaires
data['is_férié'] = data['DATETIME'].dt.date.isin(jours_feries).astype(int)

val['is_férié'] = val['DATETIME'].dt.date.isin(jours_feries).astype(int)


vacances_scolaires_2017_2023 = [
    "2017-12-23","2017-12-24","2017-12-25","2017-12-26","2017-12-27","2017-12-28","2017-12-29","2017-12-30","2017-12-31",
    "2018-01-01","2018-01-02","2018-01-03","2018-01-04","2018-01-05","2018-01-06","2018-01-07",
    "2018-02-10","2018-02-11","2018-02-12","2018-02-13","2018-02-14","2018-02-15","2018-02-16","2018-02-17","2018-02-18","2018-02-19",
    "2018-02-20","2018-02-21","2018-02-22","2018-02-23","2018-02-24","2018-02-25","2018-02-26","2018-02-27","2018-02-28","2018-03-01",
    "2018-03-02","2018-03-03","2018-03-04","2018-03-05","2018-03-06","2018-03-07","2018-03-08","2018-03-09","2018-03-10","2018-03-11",
    "2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15","2018-04-16",
    "2018-04-17","2018-04-18","2018-04-19","2018-04-20","2018-04-21","2018-04-22","2018-04-23","2018-04-24","2018-04-25","2018-04-26",
    "2018-04-27","2018-04-28","2018-04-29","2018-04-30",
    "2018-07-07","2018-07-08","2018-07-09","2018-07-10","2018-07-11","2018-07-12","2018-07-13","2018-07-14","2018-07-15","2018-07-16",
    "2018-07-17","2018-07-18","2018-07-19","2018-07-20","2018-07-21","2018-07-22","2018-07-23","2018-07-24","2018-07-25","2018-07-26",
    "2018-07-27","2018-07-28","2018-07-29","2018-07-30","2018-07-31","2018-08-01","2018-08-02","2018-08-03","2018-08-04","2018-08-05",
    "2018-08-06","2018-08-07","2018-08-08","2018-08-09","2018-08-10","2018-08-11","2018-08-12","2018-08-13","2018-08-14","2018-08-15",
    "2018-08-16","2018-08-17","2018-08-18","2018-08-19","2018-08-20","2018-08-21","2018-08-22","2018-08-23","2018-08-24","2018-08-25",
    "2018-08-26","2018-08-27","2018-08-28","2018-08-29","2018-08-30","2018-08-31","2018-09-01","2018-09-02",
    "2018-10-20","2018-10-21","2018-10-22","2018-10-23","2018-10-24","2018-10-25","2018-10-26","2018-10-27","2018-10-28","2018-10-29",
    "2018-10-30","2018-10-31","2018-11-01","2018-11-02","2018-11-03","2018-11-04","2018-11-05","2018-12-22","2018-12-23","2018-12-24","2018-12-25","2018-12-26","2018-12-27","2018-12-28","2018-12-29","2018-12-30","2018-12-31","2019-01-01","2019-01-02","2019-01-03","2019-01-04","2019-01-05","2019-01-06","2019-01-07","2019-02-09","2019-02-10","2019-02-11","2019-02-12","2019-02-13","2019-02-14","2019-02-15","2019-02-16","2019-02-17","2019-02-18",
    "2019-02-19","2019-02-20","2019-02-21","2019-02-22","2019-02-23","2019-02-24","2019-02-25","2019-02-26","2019-02-27","2019-02-28",
    "2019-03-01","2019-03-02","2019-03-03","2019-03-04","2019-03-05","2019-03-06","2019-03-07","2019-03-08","2019-03-09","2019-03-10",
    "2019-03-11",
    "2019-04-06","2019-04-07","2019-04-08","2019-04-09","2019-04-10","2019-04-11","2019-04-12","2019-04-13","2019-04-14","2019-04-15",
    "2019-04-16","2019-04-17","2019-04-18","2019-04-19","2019-04-20","2019-04-21","2019-04-22","2019-04-23","2019-04-24","2019-04-25",
    "2019-04-26","2019-04-27","2019-04-28","2019-04-29","2019-04-30","2019-05-01","2019-05-02","2019-05-03","2019-05-04","2019-05-05",
    "2019-05-06","2019-07-06","2019-07-07","2019-07-08","2019-07-09","2019-07-10","2019-07-11","2019-07-12","2019-07-13","2019-07-14","2019-07-15",
    "2019-07-16","2019-07-17","2019-07-18","2019-07-19","2019-07-20","2019-07-21","2019-07-22","2019-07-23","2019-07-24","2019-07-25",
    "2019-07-26","2019-07-27","2019-07-28","2019-07-29","2019-07-30","2019-07-31","2019-08-01","2019-08-02","2019-08-03","2019-08-04",
    "2019-08-05","2019-08-06","2019-08-07","2019-08-08","2019-08-09","2019-08-10","2019-08-11","2019-08-12","2019-08-13","2019-08-14",
    "2019-08-15","2019-08-16","2019-08-17","2019-08-18","2019-08-19","2019-08-20","2019-08-21","2019-08-22","2019-08-23","2019-08-24",
    "2019-08-25","2019-08-26","2019-08-27","2019-08-28","2019-08-29","2019-08-30","2019-08-31","2019-09-01","2019-09-02",
    "2019-10-19","2019-10-20","2019-10-21","2019-10-22","2019-10-23","2019-10-24","2019-10-25","2019-10-26","2019-10-27","2019-10-28",
    "2019-10-29","2019-10-30","2019-10-31","2019-11-01","2019-11-02","2019-11-03","2019-11-04",
    "2019-12-21","2019-12-22","2019-12-23","2019-12-24","2019-12-25","2019-12-26","2019-12-27","2019-12-28","2019-12-29","2019-12-30",
    "2019-12-31","2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05","2020-01-06",
    "2020-02-08","2020-02-09","2020-02-10","2020-02-11","2020-02-12","2020-02-13","2020-02-14","2020-02-15","2020-02-16","2020-02-17",
    "2020-02-18","2020-02-19","2020-02-20","2020-02-21","2020-02-22","2020-02-23","2020-02-24","2020-02-25","2020-02-26","2020-02-27",
    "2020-02-28","2020-02-29","2020-03-01","2020-03-02","2020-03-03","2020-03-04","2020-03-05","2020-03-06","2020-03-07","2020-03-08",
    "2020-03-09",
    "2020-04-04","2020-04-05","2020-04-06","2020-04-07","2020-04-08","2020-04-09","2020-04-10","2020-04-11","2020-04-12","2020-04-13",
    "2020-04-14","2020-04-15","2020-04-16","2020-04-17","2020-04-18","2020-04-19","2020-04-20","2020-04-21","2020-04-22","2020-04-23",
    "2020-04-24","2020-04-25","2020-04-26","2020-04-27","2020-04-28","2020-04-29","2020-04-30","2020-05-01","2020-05-02","2020-05-03",
    "2020-05-04",
    "2020-07-04","2020-07-05","2020-07-06","2020-07-07","2020-07-08","2020-07-09","2020-07-10","2020-07-11","2020-07-12","2020-07-13",
    "2020-07-14","2020-07-15","2020-07-16","2020-07-17","2020-07-18","2020-07-19","2020-07-20","2020-07-21","2020-07-22","2020-07-23",
    "2020-07-24","2020-07-25","2020-07-26","2020-07-27","2020-07-28","2020-07-29","2020-07-30","2020-07-31","2020-08-01","2020-08-02",
    "2020-08-03","2020-08-04","2020-08-05","2020-08-06","2020-08-07","2020-08-08","2020-08-09","2020-08-10","2020-08-11","2020-08-12",
    "2020-08-13","2020-08-14","2020-08-15","2020-08-16","2020-08-17","2020-08-18","2020-08-19","2020-08-20","2020-08-21","2020-08-22",
    "2020-08-23","2020-08-24","2020-08-25","2020-08-26","2020-08-27","2020-08-28","2020-08-29","2020-08-30","2020-08-31","2020-09-01",
    "2020-09-02",
    "2020-10-17","2020-10-18","2020-10-19","2020-10-20","2020-10-21","2020-10-22","2020-10-23","2020-10-24","2020-10-25","2020-10-26",
    "2020-10-27","2020-10-28","2020-10-29","2020-10-30","2020-10-31","2020-11-01","2020-11-02",
    "2020-12-19","2020-12-20","2020-12-21","2020-12-22","2020-12-23","2020-12-24","2020-12-25","2020-12-26","2020-12-27","2020-12-28",
    "2020-12-29","2020-12-30","2020-12-31","2021-01-01","2021-01-02","2021-01-03","2021-01-04",
    "2021-02-06","2021-02-07","2021-02-08","2021-02-09","2021-02-10","2021-02-11","2021-02-12","2021-02-13","2021-02-14","2021-02-15",
    "2021-02-16","2021-02-17","2021-02-18","2021-02-19","2021-02-20","2021-02-21","2021-02-22","2021-02-23","2021-02-24","2021-02-25",
    "2021-02-26","2021-02-27","2021-02-28","2021-03-01",
    "2021-04-10","2021-04-11","2021-04-12","2021-04-13","2021-04-14","2021-04-15","2021-04-16","2021-04-17","2021-04-18","2021-04-19",
    "2021-04-20","2021-04-21","2021-04-22","2021-04-23","2021-04-24","2021-04-25","2021-04-26","2021-04-27","2021-04-28","2021-04-29",
    "2021-04-30","2021-05-01","2021-05-02","2021-05-03",
    "2021-07-07","2021-07-08","2021-07-09","2021-07-10","2021-07-11","2021-07-12","2021-07-13","2021-07-14","2021-07-15","2021-07-16",
    "2021-07-17","2021-07-18","2021-07-19","2021-07-20","2021-07-21","2021-07-22","2021-07-23","2021-07-24","2021-07-25","2021-07-26",
    "2021-07-27","2021-07-28","2021-07-29","2021-07-30","2021-07-31","2021-08-01","2021-08-02","2021-08-03","2021-08-04","2021-08-05",
    "2021-08-06","2021-08-07","2021-08-08","2021-08-09","2021-08-10","2021-08-11","2021-08-12","2021-08-13","2021-08-14","2021-08-15",
    "2021-08-16","2021-08-17","2021-08-18","2021-08-19","2021-08-20","2021-08-21","2021-08-22","2021-08-23","2021-08-24","2021-08-25",
    "2021-08-26","2021-08-27","2021-08-28","2021-08-29","2021-08-30","2021-08-31","2021-09-01","2021-09-02",
    "2021-10-23","2021-10-24","2021-10-25","2021-10-26","2021-10-27","2021-10-28","2021-10-29","2021-10-30","2021-10-31","2021-11-01",
    "2021-11-02","2021-11-03","2021-11-04","2021-11-05","2021-11-06","2021-11-07","2021-11-08",
    "2021-12-18","2021-12-19","2021-12-20","2021-12-21","2021-12-22","2021-12-23","2021-12-24","2021-12-25","2021-12-26","2021-12-27",
    "2021-12-28","2021-12-29","2021-12-30","2021-12-31","2022-01-01","2022-01-02","2022-01-03",
    "2022-02-05","2022-02-06","2022-02-07","2022-02-08","2022-02-09","2022-02-10","2022-02-11","2022-02-12","2022-02-13","2022-02-14",
    "2022-02-15","2022-02-16","2022-02-17","2022-02-18","2022-02-19","2022-02-20","2022-02-21","2022-02-22","2022-02-23","2022-02-24",
    "2022-02-25","2022-02-26","2022-02-27","2022-02-28","2022-03-01","2022-03-02","2022-03-03","2022-03-04","2022-03-05","2022-03-06","2022-03-07",
    "2022-04-09","2022-04-10","2022-04-11","2022-04-12","2022-04-13","2022-04-14","2022-04-15","2022-04-16","2022-04-17","2022-04-18",
    "2022-04-19","2022-04-20","2022-04-21","2022-04-22","2022-04-23","2022-04-24","2022-04-25","2022-04-26","2022-04-27","2022-04-28",
    "2022-04-29","2022-04-30","2022-05-01","2022-05-02","2022-05-03","2022-05-04","2022-05-05","2022-05-06","2022-05-07","2022-05-08","2022-05-09",
    "2022-07-07","2022-07-08","2022-07-09","2022-07-10","2022-07-11","2022-07-12","2022-07-13","2022-07-14","2022-07-15","2022-07-16",
    "2022-07-17","2022-07-18","2022-07-19","2022-07-20","2022-07-21","2022-07-22","2022-07-23","2022-07-24","2022-07-25","2022-07-26",
    "2022-07-27","2022-07-28","2022-07-29","2022-07-30","2022-07-31","2022-08-01","2022-08-02","2022-08-03","2022-08-04","2022-08-05",
    "2022-08-06","2022-08-07","2022-08-08","2022-08-09","2022-08-10","2022-08-11","2022-08-12","2022-08-13","2022-08-14","2022-08-15",
    "2022-08-16","2022-08-17","2022-08-18","2022-08-19","2022-08-20","2022-08-21","2022-08-22","2022-08-23","2022-08-24","2022-08-25",
    "2022-08-26","2022-08-27","2022-08-28","2022-08-29","2022-08-30","2022-08-31","2022-09-01","2022-09-02",
    "2022-10-22","2022-10-23","2022-10-24","2022-10-25","2022-10-26","2022-10-27","2022-10-28","2022-10-29","2022-10-30","2022-10-31",
    "2022-11-01","2022-11-02","2022-11-03","2022-11-04","2022-11-05","2022-11-06","2022-11-07",
    "2022-12-17","2022-12-18","2022-12-19","2022-12-20","2022-12-21","2022-12-22","2022-12-23","2022-12-24","2022-12-25","2022-12-26",
    "2022-12-27","2022-12-28","2022-12-29","2022-12-30","2022-12-31","2023-01-01","2023-01-02","2023-01-03",
    "2023-02-04","2023-02-05","2023-02-06","2023-02-07","2023-02-08","2023-02-09","2023-02-10","2023-02-11","2023-02-12","2023-02-13",
    "2023-02-14","2023-02-15","2023-02-16","2023-02-17","2023-02-18","2023-02-19","2023-02-20","2023-02-21","2023-02-22","2023-02-23",
    "2023-02-24","2023-02-25","2023-02-26","2023-02-27","2023-02-28","2023-03-01","2023-03-02","2023-03-03","2023-03-04","2023-03-05","2023-03-06",
    "2023-04-08","2023-04-09","2023-04-10","2023-04-11","2023-04-12","2023-04-13","2023-04-14","2023-04-15","2023-04-16","2023-04-17",
    "2023-04-18","2023-04-19","2023-04-20","2023-04-21","2023-04-22","2023-04-23","2023-04-24","2023-04-25","2023-04-26","2023-04-27",
    "2023-04-28","2023-04-29","2023-04-30","2023-05-01","2023-05-02","2023-05-03","2023-05-04","2023-05-05","2023-05-06","2023-05-07",
    "2023-05-08","2023-05-09",
    "2023-07-08","2023-07-09","2023-07-10","2023-07-11","2023-07-12","2023-07-13","2023-07-14","2023-07-15","2023-07-16","2023-07-17",
    "2023-07-18","2023-07-19","2023-07-20","2023-07-21","2023-07-22","2023-07-23","2023-07-24","2023-07-25","2023-07-26","2023-07-27",
    "2023-07-28","2023-07-29","2023-07-30","2023-07-31","2023-08-01","2023-08-02","2023-08-03","2023-08-04","2023-08-05","2023-08-06",
    "2023-08-07","2023-08-08","2023-08-09","2023-08-10","2023-08-11","2023-08-12","2023-08-13","2023-08-14","2023-08-15","2023-08-16",
    "2023-08-17","2023-08-18","2023-08-19","2023-08-20","2023-08-21","2023-08-22","2023-08-23","2023-08-24","2023-08-25","2023-08-26",
    "2023-08-27","2023-08-28","2023-08-29","2023-08-30","2023-08-31"

]

vacances_scolaires_2017_2023_dt = pd.to_datetime(vacances_scolaires_2017_2023).date

data["vacances"] = data["DATETIME"].dt.date.isin(vacances_scolaires_2017_2023_dt).astype(int)

val["vacances"] = val["DATETIME"].dt.date.isin(vacances_scolaires_2017_2023_dt).astype(int)


#remplissage du nan dans les colonnes manquantes
fillna_mapping = {
    'CURRENT_WAIT_TIME_lag_1': const_lag15,
    'CURRENT_WAIT_TIME_lag_2': const_lag30,
    'TIME_TO_PARADE_1': const_parade1,
    'TIME_TO_PARADE_2': const_parade2,
    'TIME_TO_NIGHT_SHOW': const_paradenight,
    'snow_1h': const_snow,

}

for col in fillna_mapping:
    data[col] = data[col].fillna(fillna_mapping[col])
    val[col] = val[col].fillna(fillna_mapping[col])
    

#filtrage pour enlever les données aberrantes
data = data[ data["WAIT_TIME_IN_2H"] <=85]


# Préparation des données (on enlève les colonnes que l'on utilisera pas pour l'entrainement)
X = data.drop(columns=['WAIT_TIME_IN_2H',"DATETIME","dayofweek","month","minute",'hour',"dew_point","pressure","humidity","datetime_plus_2h","exists_minus_15","exists_minus_30","CURRENT_WAIT_TIME_lag_1","CURRENT_WAIT_TIME_lag_2","wind_speed","rain_1h","clouds_all"])
y = data['WAIT_TIME_IN_2H']

# Paramètres identiques pour tous les modèles
base_params = {
    'n_estimators': 1000,
    'max_depth': 4,
    'learning_rate': 0.04,
    'subsample': 0.8,
    'objective': 'reg:squarederror',
    'early_stopping_rounds': 20,
    'reg_alpha': 5,
    'reg_lambda': 5,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'eval_metric': 'rmse'
}

# Création de l'ensemble de modèles
models = []
train_rmses = []
val_rmses = []

#créations de 10 modèles entrainés sur des train set et validation set différents

for i in range(10):
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.15,
        random_state=42 + i  # Différent random_state pour chaque modèle
    )
    
    params = base_params.copy()
    params['random_state'] = 42 + i
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0  
    )
    
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    models.append(model)
    train_rmses.append(train_rmse)
    val_rmses.append(val_rmse)

# Statistiques de l'ensemble
print(f" RÉSULTATS DE L'ENSEMBLE :")
print(f" RMSE Train moyen: {np.mean(train_rmses):.4f} (±{np.std(train_rmses):.4f})")
print(f" RMSE Validation moyen: {np.mean(val_rmses):.4f} (±{np.std(val_rmses):.4f})")

# Fonction de prédiction d'ensemble (moyenne des 10 modèles)
def predict_ensemble(X_new):
    predictions = []
    for model in models:
        pred = model.predict(X_new)
        predictions.append(pred)

    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# Test de l'ensemble sur un échantillon de validation global
X_test_global, X_val_global, y_test_global, y_val_global = train_test_split(
    X, y, test_size=0.2, random_state=99
)

ensemble_pred = predict_ensemble(X_val_global)
ensemble_rmse = np.sqrt(mean_squared_error(y_val_global, ensemble_pred))

print(f" RMSE Ensemble global: {ensemble_rmse:.4f}")
print(f" Nombre de modèles dans l'ensemble: {len(models)}")

# Comparaison avec le meilleur modèle individuel
best_individual_rmse = min(val_rmses)
print(f" Meilleur modèle individuel: {best_individual_rmse:.4f}")
if ensemble_rmse < best_individual_rmse:
    improvement = best_individual_rmse - ensemble_rmse
    print(f" L'ensemble améliore de {improvement:.4f} RMSE !")
else:
    print(f" Ensemble: {ensemble_rmse:.4f} vs Meilleur individuel: {best_individual_rmse:.4f}")

print(f" {len(models)} modèles prêts à utiliser avec predict_ensemble(X_new)")
    
    
    
## Prédiction des temps d'atente pour le set waiting_times_X_test_final ie le validation set en ligne
# Préparer X_test_val pour la prédiction
X_test_val = val[X.columns]

#y_val_pred = xgb_final.predict(X_test_val)
y_val_pred=predict_ensemble(X_test_val)

for i in range(len(y_val_pred)):
    if y_val_pred[i]<=0:
        y_val_pred[i]=0
    if y_val_pred[i]>=85:
        y_val_pred[i]=85

submission = pd.DataFrame({
    'DATETIME': val['DATETIME'],
    'ENTITY_DESCRIPTION_SHORT': entity_desc,
    'y_pred': y_val_pred,
    'KEY': 'Validation'
})

submission.to_csv('HACKATON-ELEVEN-GROUP-11/submission_validation.csv', index=False)
    
    
