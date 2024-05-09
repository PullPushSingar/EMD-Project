import numpy as np
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from emd import emd

CITY_NAME = 'Chicago'

temperature_df = pd.read_csv('Data/temperature.csv', parse_dates=['datetime'])
temperature_df.set_index('datetime', inplace=True)

# print(temperature_df['New York'].head())

# missing_values = temperature_df.isnull().sum()
# print(missing_values)

# forward-filling all missing values
temperature_df.ffill(inplace=True)
# backward-filling to fill first missing value
temperature_df.bfill(inplace=True)

# check if no NaN values remain
assert not temperature_df.isnull().any().any(), "There are still NaN values in dataset"

# print(temperature_df['New York'].head())

plt.plot(temperature_df[CITY_NAME], label=f'Temperature {CITY_NAME}')
plt.title(f'Hourly Temperature in {CITY_NAME}')
plt.xlabel('Date')
plt.ylabel('Temperature (Kelvin)')
plt.legend()
plt.show()
      
emd(np.array(temperature_df[CITY_NAME]),1, 5, True)
