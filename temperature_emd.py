import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
      
imfs = emd(np.array(temperature_df[CITY_NAME]),1, 5, True)
print(imfs[4])

# temperature = temperature_df[[CITY_NAME]].values
# print(temperature_df[[CITY_NAME]].values)

temperature_values = [[temp] for temp in imfs[4]]
temperature = np.array(temperature_values)

# Normalizing 
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_scaled = scaler.fit_transform(temperature)

# Splitting the data into training and testing sets
train_size = int(len(temperature_scaled) * 0.8)
test_size = len(temperature_scaled) - train_size
train, test = temperature_scaled[0:train_size,:], temperature_scaled[train_size:len(temperature_scaled),:]


# Shaping data to be suitable for LSTM

# Converting an array of values into a dataset matrix for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 24
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Input will be: [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out_last_step = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out_last_step)
        return predictions

input_size = 1
hidden_layer_size = 50
num_layers = 1
output_size = 1

model = LSTMModel(input_size, hidden_layer_size, num_layers, output_size)
# Defining the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


print(model)


X_train_tensor = torch.Tensor(X_train)
Y_train_tensor = torch.Tensor(Y_train).view(-1, 1)

train_data = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)



# Training
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for i in range(epochs):
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = criterion(y_pred, labels)
        single_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

    if i % 5 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {epochs:3} loss: {single_loss.item():10.10f}')

# Evaluation
X_test_tensor = torch.Tensor(X_test)
Y_test_tensor = torch.Tensor(Y_test).view(-1, 1)

# Create DataLoader for testing data
test_data = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

actuals = np.array([[label.item()] for label in Y_test_tensor])
actuals = scaler.inverse_transform(actuals)
predictions = []

model.eval()
with torch.no_grad():
    for seq, labels in test_loader:
        seq = seq.to(device)
        y_pred = model(seq).cpu().numpy()
        predictions.append(y_pred[0][0])

predictions = [[temp] for temp in predictions]
predictions_arr = np.array(predictions)
prediction_inverrse = scaler.inverse_transform(predictions_arr)

# Visualize predictions vs actuals
plt.figure(figsize=(15, 5))
plt.plot(actuals, label='Actual Values')
plt.plot(prediction_inverrse, label='Predictions', alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()

