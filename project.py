import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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



def main():
    path = 'PRSA_Data_Aotizhongxin_20130301-20170228.csv'

    try:
        df = load_air_quality_data(path)
        df = preprocess_missing_values(df)
        print(f"Podaci uspjesno ucitani. Ukupan broj uzoraka: {len(df)}")

        if df['PM2.5'].isnull().sum() == 0:
            print("Nema nedostajucih vrijednosti.")

        print("\nPregled obradjenih podataka:")
        print(df.head())
    
    except FileNotFoundError:
        print(f"Greska: Fajl nije pronadjen.")
    except Exception as e:
        print(f"Doslo je do neocekivane greske: {e}")

if __name__ == "__main__":
    main()