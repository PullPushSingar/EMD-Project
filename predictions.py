import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

def get_data(data_types: list) -> dict[pd.DataFrame]:
    '''Gets data from csv's in ./Data/ directory'''
    df_dict = {}
    for data_type in tqdm(data_types, desc="Data from csv"):
        # Make sure the csv data files exist
        if not os.path.isfile(f'Data/{data_type}.csv'):
            continue
        df = pd.read_csv(f'Data/{data_type}.csv', parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)

        # forward-filling all missing values
        df.ffill(inplace=True)
        # backward-filling to fill first missing value
        df.bfill(inplace=True)

        # check if no NaN values remain
        assert not df.isnull().any().any(), "There are still NaN values in dataset"
        
        df_dict[data_type] = df
    return df_dict

def list_by_cities(data, data_types):
    '''Reshapes data from list data_type(city(value)) to dict {city:(data_type(value))}'''
    cities = list(data[data_types[0]])
    cities_dict = {}
    for city in tqdm(cities, desc="List for each city"):
        # Merge the three DataFrames on datetime
        city_df = data[data_types[0]][[city]]
        for i in range(1, len(data_types)):
            city_df = city_df.merge(data[data_types[i]][[city]], left_index=True, right_index=True)
        
        # Rename columns
        city_df.columns = data_types
        
        # Append to the array
        cities_dict[city] = city_df
    return cities_dict

def df_to_X_y(df_dict: dict[pd.DataFrame], window_size: int):
    '''Converts DataFrames to (x y) sets'''
    df_as_np_dict = {key: df.to_numpy() for key, df in df_dict.items()}

    x_y_by_cities = {}
    for city, npdf in tqdm(df_as_np_dict.items(), desc="Df to X,Y"):
        num_rows = len(npdf)
        # Preallocate arrays for X and y
        X = np.zeros((num_rows - window_size, window_size, npdf.shape[1]))
        y = np.zeros((num_rows - window_size,))

        for i in range(num_rows - window_size):
            # Use slicing for faster array creation
            X[i] = npdf[i:i+window_size]
            y[i] = npdf[i+window_size][-1]
            
        x_y_by_cities[city] = (X, y)
    
    return x_y_by_cities

def imf_to_X_y(imf: list[list[int]], window_size: int):
    '''Converts imf to (x y) sets'''
    x = []
    y = [] 

    imf_np = np.array(imf)

    num_rows = len(imf)
    # Preallocate arrays for X and y
    x = np.zeros((num_rows - window_size, window_size, imf_np.shape[1]))
    y = np.zeros((num_rows - window_size,))

    for i in range(num_rows - window_size):
        # Use slicing for faster array creation
        x[i] = imf_np[i:i+window_size]
        y[i] = imf_np[i+window_size][-1]
    
    return np.array(x), np.array(y)

def get_x_y_from_cities(data, cities=None):
    '''If cities == None then get all data'''
    x = []
    y = []
    for city_name, city_x_y in tqdm(data.items(), desc="Merge city data"):
        if cities == None or city_name in cities:
            for temp_x in city_x_y[0]:
                x.append(temp_x.copy())
            for temp_y in city_x_y[1]:
                y.append(temp_y.copy())
    return np.array(x), np.array(y)

def get_xy_from_csv(data_types, window_size):
    "Returns X and Y sets"
    data = get_data(data_types)
    cities_dict = list_by_cities(data, data_types)
    return df_to_X_y(cities_dict, window_size)


def divide_data(x, y, train_pc, val_pc):
    '''Divides data into training, validation and test sets'''
    assert train_pc + val_pc < 1
    train_len = int(train_pc * len(x))
    val_i = int((train_pc + val_pc)* len(x))
    x_train, y_train = x[:train_len], y[:train_len]
    x_val, y_val = x[train_len:val_i], y[train_len:val_i]
    x_test, y_test = x[val_i:], y[val_i:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def run_lstm_model(model_name, x_train, y_train, x_val, y_val, window_size, data_types, epochs=10, learning_rate=0.0001):
    model = Sequential()
    model.add(InputLayer((window_size, len(data_types))))
    model.add(LSTM(64))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, 'linear'))

    # Save best fit models to model/ directory
    cp = ModelCheckpoint(f'model/lstm/{model_name}.keras', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    model.summary()

    # Training
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, callbacks=[cp])

def lstm_predict(model_name, x):
    model = load_model(f'model/lstm/{model_name}.keras')
    return model.predict(x).flatten()

def as_yesterday_predict(x):
    assert all(len(item) >= 5 for item in x)
    return [vals[-5][-1] for vals in x]

def as_last_hour_predict(x):
    return [vals[-1][-1] for vals in x]