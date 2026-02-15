import numpy as np
import pandas as pd

### processing the weather data



df = pd.read_csv("weather_data.csv")
df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True)
df = df.sort_values("DATETIME")

# Colonnes météo de base (adapte les noms si besoin)
base_cols = ["DATETIME", "rain_1h"]

# Copie décalée: la dfeur mesurée à t (rain_1h, snow_1h) sera alignée sur t-2h
fut = df[base_cols].copy()
fut["DATETIME"] = fut["DATETIME"] - pd.Timedelta(hours=2)
fut = fut.rename(columns={
    "rain_1h": "rain_in_2h",
    "snow_1h": "snow_in_2h"
})

# Merge 1:1 sur la DATETIME “courante”
df = df.merge(fut, on="DATETIME", how="left", validate="1:1")

df['DATETIME'] = pd.to_datetime(df['DATETIME'])

#df['dayofweek'] = df['DATETIME'].dt.dayofweek

df['minute'] = df['DATETIME'].dt.hour*60 + df['DATETIME'].dt.minute
df['month'] = df['DATETIME'].dt.month



df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 1440)
df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 1440)

print(df.shape)




### select features and target to train model 

# Garder uniquement les lignes avec une dfeur connue de pluie dans 2h
df_train = df[df["rain_in_2h"].notna()].copy()
print(df_train.shape)

df_train.to_csv("weather_prediction_data.csv", index=False)