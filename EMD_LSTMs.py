from predictions import *
from emd import emd

def trainig_oneLSTM():
    data_types = ["temperature"] 
    window_size = 5

    data = get_data(data_types)
    cities_dict = list_by_cities(data, data_types)

    x_y_by_cities= df_to_X_y(cities_dict,window_size)
    x, y = get_x_y_from_cities(x_y_by_cities, ["Chicago"])
    x_train, y_train, x_val, y_val, x_test, y_test = divide_data(x, y , train_pc=0.8, val_pc=0.1)
    
    model_name = "chicago_5h_temp_only"
    learning_rate = 0.0001
    num_of_epochs = 100
    run_lstm_model(model_name, x_train, y_train, x_val, y_val, window_size, data_types, epochs=num_of_epochs, learning_rate=learning_rate)
    lstm_prediction = lstm_predict(model_name, x_test)

    ay_prediction = as_yesterday_predict(x_test)
    alh_prediction = as_last_hour_predict(x_test)

    plt.title("one LSTM")
    plt.plot(y_test[50: 50 + 24*7], "b")
    plt.plot(lstm_prediction[50: 50 + 24*7], "g")
    plt.plot(ay_prediction[50: 50 + 24*7], "r")
    plt.plot(alh_prediction[50: 50 + 24*7], "m")
    plt.legend(["Actual", "one LSTM", "As Yesterday", "As Last Hour"])
    plt.draw()
    plt.show()

    print(f"MSE(lstm): {mse(y_test, lstm_prediction)}")
    print(f"MSE(ay): {mse(y_test, ay_prediction)}")
    print(f"MSE(alh): {mse(y_test, alh_prediction)}")


def training():
    numberIMFs = 5
    data_types = ["temperature"] 
    window_size = 5

    data = get_data(data_types)
    imfs, signal = emd(np.array(data["temperature"]["Chicago"]), 1, numberIMFs, True)

    for imf_iter in range(numberIMFs):
        imf = [[temp] for temp in imfs[imf_iter]]

        x, y = imf_to_X_y(imf,window_size)
        x_train, y_train, x_val, y_val, x_test, y_test = divide_data(x, y , train_pc=0.8, val_pc=0.1)

        model_name = f"emdlstm/chicago_emd_imf{imf_iter}"
        learning_rate = 0.0001
        num_of_epochs = 100
        # run_lstm_model(model_name, x_train, y_train, x_val, y_val, window_size, data_types, epochs=num_of_epochs, learning_rate=learning_rate)
        lstm_prediction = lstm_predict(model_name, x_test)

        x_as, y_as = imf_to_X_y(imf,24)
        x_train_as, y_train_as, x_val_as, y_val_as, x_test_as, y_test_as = divide_data(x_as, y_as, train_pc=0.8, val_pc=0.1)
        ay_prediction = as_yesterday_predict(x_test_as)
        alh_prediction = as_last_hour_predict(x_test)

        plt.title(f"IMF {imf_iter}")
        plt.plot(y_test[50: 50 + 24*7], "b")
        plt.plot(lstm_prediction[50: 50 + 24*7], "g")
        plt.plot(ay_prediction[50: 50 + 24*7], "r")
        plt.plot(alh_prediction[50: 50 + 24*7], "m")
        plt.legend(["Actual", "LSTM", "As Yesterday", "As Last Hour"])
        plt.draw()
        plt.show()

        print(f"MSE(lstm): {mse(y_test, lstm_prediction)}")
        print(f"MSE(ay): {mse(y_test_as, ay_prediction)}")
        print(f"MSE(alh): {mse(y_test, alh_prediction)}")

    signal_n = [[temp] for temp in signal]

    x, y = imf_to_X_y(signal_n,window_size)
    x_train, y_train, x_val, y_val, x_test, y_test = divide_data(x, y , train_pc=0.8, val_pc=0.1)

    model_name = f"emdlstm/residual"
    learning_rate = 0.0001
    num_of_epochs = 100
    # run_lstm_model(model_name, x_train, y_train, x_val, y_val, window_size, data_types, epochs=num_of_epochs, learning_rate=learning_rate)
    lstm_prediction = lstm_predict(model_name, x_test)

    x_as, y_as = imf_to_X_y(signal_n,24)
    x_train_as, y_train_as, x_val_as, y_val_as, x_test_as, y_test_as = divide_data(x_as, y_as, train_pc=0.8, val_pc=0.1)
    ay_prediction = as_yesterday_predict(x_test_as)
    alh_prediction = as_last_hour_predict(x_test)

    plt.title(f"residual")
    plt.plot(y_test[50: 50 + 24*7], "b")
    plt.plot(lstm_prediction[50: 50 + 24*7], "g")
    plt.plot(ay_prediction[50: 50 + 24*7], "r")
    plt.plot(alh_prediction[50: 50 + 24*7], "m")
    plt.legend(["Actual", "LSTM", "As Yesterday", "As Last Hour"])
    plt.draw()
    plt.show()

    print(f"MSE(lstm): {mse(y_test, lstm_prediction)}")
    print(f"MSE(ay): {mse(y_test_as, ay_prediction)}")
    print(f"MSE(alh): {mse(y_test, alh_prediction)}")


def predict():
    numberIMFs = 5
    data_types = ["temperature"] 
    window_size = 5
    predicted_imss = []

    data = get_data(data_types)

    cities_dict = list_by_cities(data, data_types)
    x_y_by_cities= df_to_X_y(cities_dict,window_size)
    x, y = get_x_y_from_cities(x_y_by_cities, ["Chicago"])
    x_train_all, y_train_all, x_val_all, y_val_all, x_test_all, y_test_all = \
        divide_data(x, y , train_pc=0.8, val_pc=0.1)
    
    imfs, signal = emd(np.array(data["temperature"]["Chicago"]), 1, numberIMFs, True)

    for imf_iter in range(numberIMFs):
        imf = [[temp] for temp in imfs[imf_iter]]
        x, y = imf_to_X_y(imf,window_size)
        x_train, y_train, x_val, y_val, x_test, y_test = divide_data(x, y , train_pc=0.8, val_pc=0.1)

        model_name = f"emdlstm/chicago_emd_imf{imf_iter}"
        predict_imf = lstm_predict(model_name, x_test)
        print(predict_imf)
        predicted_imss.append(predict_imf)

    #residual
    signal_n = [[temp] for temp in signal]
    x, y = imf_to_X_y(signal_n,window_size)
    x_train, y_train, x_val, y_val, x_test, y_test = divide_data(x, y , train_pc=0.8, val_pc=0.1)

    model_name = f"emdlstm/residual"
    predict_residual= lstm_predict(model_name, x_test)
    predicted_imss.append(predict_residual)

    prediction = np.sum(predicted_imss, axis=0)
    print(prediction)

    x_y_by_cities24= df_to_X_y(cities_dict,24)
    x_as, y_as = get_x_y_from_cities(x_y_by_cities24, ["Chicago"])
    x_train_as, y_train_as, x_val_as, y_val_as, x_test_as, y_test_as = divide_data(x_as, y_as, train_pc=0.8, val_pc=0.1)
    ay_prediction = as_yesterday_predict(x_test_as)
    alh_prediction = as_last_hour_predict(x_test_all)

    model_name = "lstm/chicago_5h_temp_only"
    one_LSTM_prediction = lstm_predict(model_name, x_test_all)

    plt.plot(y_test_all[50: 50 + 24*7], "b")
    plt.plot(prediction[50: 50 + 24*7], "g")
    plt.plot(ay_prediction[50: 50 + 24*7], "r")
    plt.plot(alh_prediction[50: 50 + 24*7], "m")
    plt.legend(["Actual", "EMD+LSTM", "As Yesterday", "As Last Hour"])
    plt.draw()
    plt.show()

    plt.plot(y_test_all[50: 50 + 24*7], "b")
    plt.plot(prediction[50: 50 + 24*7], "g")
    plt.plot(one_LSTM_prediction[50: 50 + 24*7], "r")
    plt.legend(["Actual", "EMD+LSTM", "one LSTM"])
    plt.draw()
    plt.show()

    print(f"MSE(EMD+lstm): {mse(y_test_all, prediction)}")
    print(f"MSE(ay): {mse(x_test_as, ay_prediction)}")
    print(f"MSE(alh): {mse(y_test_all, alh_prediction)}")
    print(f"MSE(oneLSTM): {mse(y_test_all, one_LSTM_prediction)}")

def mse(a,b):
    return (np.square(a - b)).mean()

if __name__ == "__main__":
    predict()