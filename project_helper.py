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

def run_arima_model(df):
    print("\n--- Pokretanje ARIMA modela ---")
    
    test_size = 4320 
    train = df['PM2.5'].iloc[:-test_size]
    test = df['PM2.5'].iloc[-test_size:]
    
    # p=1 (gleda sat unazad), d=1 (gleda promjenu), q=1 (ispravlja gresku)
    # koristimo metodu 'innovations_mle' koja je brza za slabije procesore
    model = ARIMA(train, order=(1, 1, 1))
    
    # Dodajemo maxiter=20 da ne bi vrtio u beskonacno ako ne moze da nadje rjesenje
    model_fit = model.fit(method='innovations_mle') 
    
    # Predviđanje
    forecast = model_fit.forecast(steps=24)
    
    # RACUNANJE METRIKA (Sekcija 5 tvog rada)
    rmse = np.sqrt(mean_squared_error(test.iloc[:24], forecast))
    mae = mean_absolute_error(test.iloc[:24], forecast)
    
    print(f"ARIMA RMSE: {rmse:.2f}")
    print(f"ARIMA MAE: {mae:.2f}")

    # Grafik
    plt.figure(figsize=(10,5))
    plt.plot(test.iloc[:24].values, label='Stvarne vrijednosti (Test)')
    plt.plot(forecast.values, label='ARIMA predikcija', color='red', linestyle='--')
    plt.title('ARIMA Benchmark: Predviđanje za 24h')
    plt.legend()
    plt.show()

def run_prophet_model(df):
    print("\n--- Pokretanje Prophet modela ---")
    
    # Prophet zahtijeva specifičan format: ds (datumi) i y (vrijednosti)
    df_prophet = df.reset_index()[['date', 'PM2.5']]
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None) # Uklanjanje vremenske zone za svaki slucaj

    test_size = 4320
    train = df_prophet.iloc[:-test_size]
    test = df_prophet.iloc[-test_size:]

    # Model sa dnevnom, sedmicnom i godisnjom sezonalnoscu
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train)

    # Predviđanje za 24h
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    
    y_pred = forecast['yhat'].tail(24).values
    y_true = test['y'].iloc[:24].values

    # Metrike
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Prophet RMSE: {rmse:.2f}")
    print(f"Prophet MAE: {mae:.2f}")

    # Grafik
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Stvarne vrijednosti')
    plt.plot(y_pred, label='Prophet predikcija', color='green')
    plt.title('Prophet: Predviđanje za 24h')
    plt.legend()
    plt.show()

def run_lstm_model(df):
    print("\n--- Pokretanje LSTM modela ---")
    
    # Skaliranje podataka (bitno za neuronske mreze)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['PM2.5']])
    
    # Parametri
    look_back = 24 # Model gleda zadnja 24 sata da bi predvidio sljedeci
    test_size = 4320
    
    # Priprema podataka za LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            X.append(dataset[i:(i+look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    train_data = data_scaled[:-test_size]
    test_data = data_scaled[-test_size-look_back:]

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    # Reshape za LSTM: [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Arhitektura mreze
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Trening (na Colabu ce trajati kratko)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Predviđanje
    predictions = model.predict(X_test[:24], verbose=0)
    predictions = scaler.inverse_transform(predictions) # Vracanje u originalne vrijednosti
    
    y_true_original = df['PM2.5'].iloc[-test_size : -test_size + 24].values

    # Metrike
    rmse = np.sqrt(mean_squared_error(y_true_original, predictions))
    print(f"LSTM RMSE: {rmse:.2f}")

    # Grafik
    plt.figure(figsize=(10,5))
    plt.plot(y_true_original, label='Stvarne vrijednosti')
    plt.plot(predictions, label='LSTM predikcija', color='orange')
    plt.title('LSTM: Predviđanje za 24h')
    plt.legend()
    plt.show()