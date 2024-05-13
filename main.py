from predictions import *
from emd import emd

def main():
    # Variables
    # Temperature must be last in data_types
    data_types = ["humidity", "pressure", "temperature"] # Data used for prediction
    window_size = 24

    data = get_data(data_types)
    cities_dict = list_by_cities(data, data_types)

    # x_y_by_cities is a dict with city names as keys
    # and values are tuples (x, y)
    x_y_by_cities= df_to_X_y(cities_dict,window_size)
    x, y = get_x_y_from_cities(x_y_by_cities, ["Kansas City", "Saint Louis"])
    
    # Divide the data
    x_train, y_train, x_val, y_val, x_test, y_test = \
        divide_data(x, y , train_pc=0.8, val_pc=0.1)
    
    # # model_name saved in ./model/ dir
    # # !!! run_lstm_model() overwrites the older model with the same name
    model_name = "chicago_24h"
    learning_rate = 0.0001
    num_of_epochs = 10
    # run_lstm_model(model_name, x_train, y_train, x_val, y_val, window_size, data_types, epochs=num_of_epochs, learning_rate=learning_rate)
    lstm_prediction = lstm_predict(model_name, x_test)

    ay_prediction = as_yesterday_predict(x_test)
    alh_prediction = as_last_hour_predict(x_test)

    plt.plot(y_test[50: 50 + 24*7], "b")
    plt.plot(lstm_prediction[50: 50 + 24*7], "g")
    plt.plot(ay_prediction[50: 50 + 24*7], "r")
    plt.plot(alh_prediction[50: 50 + 24*7], "m")
    plt.legend(["Actual", "LSTM", "As Yesterday", "As Last Hour"])
    plt.draw()

    print(f"MSE(lstm): {mse(y_test, lstm_prediction)}")
    print(f"MSE(ay): {mse(y_test, ay_prediction)}")
    print(f"MSE(alh): {mse(y_test, alh_prediction)}")
    
    future_lstm_prediction = future_lstm_predict(model_name, x_test[500:523],7*24)
    plt.plot(y_test[500: 523 + 24*7], "b")
    plt.plot(future_lstm_prediction, "g")
    plt.draw()

def mse(a,b):
    return (np.square(a - b)).mean()

if __name__ == "__main__":
    main()