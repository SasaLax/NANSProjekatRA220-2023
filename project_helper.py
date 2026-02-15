import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from prophet import Prophet

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')


def load_air_quality_data(file_path):
    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('date', inplace=True)

    # Fokus na kljucne kolone
    relevant_cols = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    df = df[relevant_cols]

    return df

def preprocess_missing_values(df):
    # limit 2 zbog interpolatuje ako je jedan ili dva NaN
    df['PM2.5'] = df['PM2.5'].interpolate(method='linear',limit=2)

    df['month'] = df.index.month
    df['hour'] = df.index.hour

    #grupise po istim datumima i bira srednju vrijednost za NaN
    df['PM2.5'] = df['PM2.5'].fillna(df.groupby(['month','hour'])['PM2.5'].transform('mean'))

    df.drop(['month','hour'], axis=1, inplace=True)

    return df

# odgovorice koji meteo faktor ima najjacu korelaciju sa PM2.5
def run_eda_correlation(df):
    print("\n---Pokretanje analize korelacije--\n")
    plt.figure(figsize=(10,8))

    corr_matrix = df.corr()

    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelacija PM2.5 cestica sa meteorolskim faktorima")
    plt.show()

def run_stl_decomposition(df):
    print("\n--- Pokretanje STL dekompozicije")

    #uzimamo manji uzorak zbog preglednosti grafika
    subset = df['PM2.5'].tail(1000)

    #period=24 jer podaci na dnevnom nivou(dnevna sezonalnost)
    res = STL(subset, period=24).fit()

    fig = res.plot()
    plt.suptitle('STL Dekompozicija: Trend, Sezonalnost i Reziduali', fontsize=15)
    plt.show()