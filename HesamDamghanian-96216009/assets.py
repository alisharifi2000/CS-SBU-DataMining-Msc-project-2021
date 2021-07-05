import ta
import pandas as pd

TIME_STEPS = 24


def get_sklearn_regression(regressors, x_train, y_train, x_test, y_test, graph=False, print_results=False):
    from sklearn.metrics import mean_squared_error

    trained_regressors = []
    for regressor in regressors:
        print(f'\nRegressor: {regressor().__class__.__name__}')
        reg = regressor()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        trained_regressors.append(reg)

        _ = regression_performance(pred, y_test, print_results=print_results)

        # print(f'\trmse: {rmse}')
        # print(f'\tPerformance: {performance}')
        # print(f'\tPrecision: {precision:.2f}%')

        if graph:
            prediction_graph(pred, y_test.to_numpy(),
                             model_name=regressor().__class__.__name__)
            error_graph(pred, y_test.to_numpy(),
                        model_name=regressor().__class__.__name__)

    return trained_regressors


def regression_performance(y_pred, y_true, print_results=False):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    assert len(y_pred) == len(y_true)
    res = 0
    for i, j in zip(y_true, y_pred):
        if 0.95*i <= j <= 1.05*i:
            res += 1
    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

    current = np.roll(y_true, -1)
    target = np.where(y_true-current > 0, 1, 0)
    target_pred = np.where(y_pred-current > 0, 1, 0)
    precision = 100*sum(np.equal(target, target_pred) +
                        np.zeros(len(y_true)))/len(y_true)
    performance = res/len(y_pred)
    if print_results:
        print(f'RMSE: {rmse:.2f}')
        print(f'Performance Percentage: {performance:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'R2 Score Test = {r2_score(y_true, y_pred): 0.3f}')
        print(
            f'MAE Test = {mean_absolute_error(y_true=y_true, y_pred=y_pred): 0.3f}')

    return performance, rmse, precision


def prediction_graph(y_pred, y_true, model_name=None, save_name=False):
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(y_pred)
    plt.plot(y_true)
    title = ' Prediction vs Real Price'
    if model_name:
        title = model_name + title
    plt.title(title)
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Prediction', 'Real'], loc='upper left')
    plt.grid()
    plt.show()
    if save_name:
        plt.savefig(save_name)


def bagging_regerssion(regressors, x_train, y_train, x_test, y_test, graph=False, return_preds=True):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.metrics import mean_squared_error
    import pandas as pd

    if return_preds:
        preds = pd.DataFrame()

    for regressor in regressors:
        print(f'Regressor: {regressor.__class__.__name__}')
        br = BaggingRegressor(base_estimator=regressor,
                              n_estimators=10, random_state=0)
        br.fit(x_train, y_train)
        pred = br.predict(x_test)
        performance, rmse, precision = regression_performance(
            pred, y_test.to_numpy(), print_results=True)

        if graph:
            prediction_graph(pred, y_test.to_numpy(),
                             regressor.__class__.__name__)
            error_graph(pred, y_test.to_numpy(), regressor.__class__.__name__)
        if return_preds:
            preds[f'{regressor.__class__.__name__}'] = pred
    if return_preds:
        return preds


def plot_corr_matrix(df):
    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(1, 15))
    plt.matshow(df.corr(), fignum=f.number)
    # plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(
    #     ['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(
        ['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


def error_graph(y_pred, y_test, model_name):
    import seaborn as sns
    import matplotlib.pyplot as plt
    errors = y_pred - y_test
    errors = errors.flatten()
    errors_mean = errors.mean()
    errors_std = errors.std()

    sns_c = sns.color_palette(palette='deep')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.regplot(x=y_test.flatten(), y=y_pred.flatten(), ax=ax[0])
    sns.distplot(a=errors, ax=ax[1])
    ax[1].axvline(x=errors_mean, color=sns_c[3],
                  linestyle='--', label=f'$\mu$')
    ax[1].axvline(x=errors_mean + 2*errors_std, color=sns_c[4],
                  linestyle='--', label=f'$\mu \pm 2\sigma$')
    ax[1].axvline(x=errors_mean - 2*errors_std, color=sns_c[4], linestyle='--')
    ax[1].axvline(x=errors_mean, color=sns_c[3], linestyle='--')
    ax[1].legend()
    ax[0].set(title=f'{model_name} Test vs Predictions (Test Set)',
              xlabel='y_test', ylabel='y_pred')
    ax[1].set(title=f'{model_name} Errors', xlabel='error', ylabel=None)
    plt.show()


def build_timeseries(mat, y_col_index, time_steps=TIME_STEPS):
    import numpy as np

    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]

    return x, y


class Feature_Extractor:

    # recieves data sheet as df and adds bollinger band related columns to the df.
    # frame value should be in [20, 40] (#2 TimeFrames)
    def bollingerBand(data, features, frame):
        '''
            Bollinger Bands consist of a middle band with two outer bands. The middle band is a simple moving average that is usually set at 20 periods.
        A simple moving average is used because the standard deviation formula also uses a simple moving average. 
        The look-back period for the standard deviation is the same as for the simple moving average. The outer bands are usually set 2 standard deviations
        above and below the middle band.
        '''

        bb = ta.volatility.BollingerBands(
            close=data['Close'], window=frame, window_dev=2, fillna=True)

        # Bollinger Channel Middle Band
        features[f'Bb-m-{frame} frame'] = bb.bollinger_mavg()

        # Bollinger Channel High Band
        features[f'Bb-h-{frame} frame'] = bb.bollinger_hband()

        # Bollinger Channel Low Band
        features[f'Bb-l-{frame} frame'] = bb.bollinger_lband()

        # It returns 1, if close is higher than bollinger_hband. Else, it returns 0.
        features[f'Bb-h-i-{frame} frame'] = bb.bollinger_hband_indicator()

        # It returns 1, if close is lower than bollinger_lband. Else, it returns 0.
        features[f'Bb-l-i-{frame} frame'] = bb.bollinger_lband_indicator()

    # recieves data sheet as df and adds atr column to the df.
    # frame value should be chosen from [7, 14, 21] (#3 TimeFrames)

    def atr(data, features, frame):
        '''
            The indicator provide an indication of the degree of price volatility. Strong moves, in either direction, are often accompanied by
            large ranges, or large True Ranges.
        '''

        atr = ta.volatility.AverageTrueRange(
            high=data['High'], low=data['Low'], close=data['Close'], window=frame, fillna=True)

        # Average True Range (ATR)
        features[f'Atr-{frame} frame'] = atr.average_true_range()

    # recieves data sheet as df and adds highest-{frame} column to the df.
    # frame value should be in [5, 15, 25] (#3 TimeFrames)

    def highest(data, features, frame, ohlc='Close'):
        '''
            simply adds highest value of the last three frames to the "Highest" column.
        '''
        highest = []
        for i in range(len(data)):
            if i < frame:
                highest.append(max(data[ohlc][:i+1]))
            else:
                highest.append(max(data[ohlc][i-frame:i+1]))
        # Highest
        features[f'Highest-{frame} frame'] = highest

    # recieves data sheet as df and adds howest-{frame} column to the df.
    # frame value should be in [5, 15, 25] (#3 TimeFrames)

    def lowest(data, features, frame, ohlc='Close'):
        '''
            simply adds lowest value of the last three frames to the "Highest" column.
        '''
        lowest = []
        for i in range(len(data)):
            if i < frame:
                lowest.append(min(data[ohlc][:i+1]))
            else:
                lowest.append(min(data[ohlc][i-frame:i+1]))
        # Highest
        features[f'Lowest-{frame} frame'] = lowest

    # recieves data sheet as df and adds Ma column to the df.
    # frame value should be in [M1, M5, M15, H1] (#4 TimeFrames)

    def ma(data, features, frame, ohlc):
        '''
            SMA - Simple Moving Average
        takes average of 'frame' number of previous values
        '''
        sma = ta.trend.SMAIndicator(
            close=data['Close'], window=frame, fillna=True)

        # Simple Moving Average
        features[f'Ma-{frame} frame'] = sma.sma_indicator()

    # recieves data sheet as df and adds Rsi column to the df.
    # frame value should be in [7, 14] (#? TimeFrames)

    def rsi(data, features, frame, ohlc):
        '''
            Relative Strength Index (RSI)
        Compares the magnitude of recent gains and losses over a specified time period to measure speed and change of price movements of a
        security. It is primarily used to attempt to identify overbought or oversold conditions in the trading of an asset.
        '''
        rsi = ta.momentum.RSIIndicator(
            close=data['Close'], window=frame, fillna=True)

        # Simple Moving Average
        features[f'Rsi-{frame} frame'] = rsi.rsi()

    def ohlc(data, features):
        '''
        '''

        features['Open'] = features['Open']
        features['High'] = features['High']
        features['Low'] = features['Low']
        features['Close'] = features['Close']


def save_data(data, features, name):
    '''
        save dataframe objects as csv to /out dir with given name
    for further processing
    '''
    data.to_csv(f"out\{name}.csv")


def generate_label(data, features, ohlc):
    '''
        assigns 0 for an entry with lower value, and 1 for equal or higher
    value than the latest entry. appends a column containing according labels
    to the data
    '''

    labels = []

    for value in range(len(data[ohlc])-1):
        if data[ohlc][value] < data[ohlc][value+1]:
            labels.append(1)
        else:
            labels.append(0)
    labels.append(0)
    features[f"labels-{ohlc}-based"] = labels

# maybe deleted


def generate_features(data_list):
    import Feature_Extractor as fe

    for i in data_list:
        res = pd.DataFrame()
        res_name = ''
        if i == 'data_H1':
            i = data_H1
            res_name = 'data_H1'
            fe.rsi(i, res, 14, 'Close')
            fe.rsi(i, res, 7, 'Close')
            fe.ma(i, res, 6, 'Close')
            fe.ma(i, res, 12, 'Close')
        elif i == data_M15:
            res_name = 'data_M15'
            fe.rsi(i, res, 14, 'Close')
            fe.rsi(i, res, 7, 'Close')
        elif i == data_M1:
            res_name = 'data_M1'
            for j in [20, 50, 100, 200]:
                fe.ma(i, res, j, 'Close')
            for j in [5, 15, 25]:
                fe.highest(i, res, j)
                fe.lowest(i, res, j)
        elif i == data_M1:
            res_name = 'data_M5'
            for j in [7, 14, 21]:
                fe.atr(i, res, j)
            for j in [5, 15, 25]:
                fe.highest(i, res, j)
                fe.lowest(i, res, j)
        res['Local time'] = i['Local time']
        res.to_csv(f"out\{res_name}.csv")

    combined_features = pd.DataFrame()
    combined_features['Local time'] = data_list[-1]['Local time'].to_list()

    # for date in data_list[-1]['Local time'].to_list():
    #     for iterator in os.asndsibgvab:
    #         for column in iterator.columns.to_list().remove('Local time'):

    # data_H1.index[data_H1['Local time'] == '29.08.2020 23:30:00.000 GMT+0430'].tolist()


def concat_features(path, path_to_save, time):
    '''
        concats all features given in path list to a single
    csv file containing all features (path_to_save) indexed by 
    "Local time" column.
    '''

    import pandas as pd

    features_list = []

    for i in range(len(path)):
        features_list.append(pd.read_csv(path[i]))

    result = pd.concat(features_list, axis=1, join='inner')
    result.index = result.iloc[:, 0]
    result = result.drop(time, axis=1)
    result.to_csv(path_to_save)


def append_timeframe(path, prefix, time):
    '''
        appends prefix to all columns, and sets Local time as index
    and overwrites the given path
    '''

    import pandas as pd

    m1 = pd.read_csv(path)
    m1.index = m1[time]
    m1.drop(['Unnamed: 0', time], axis=1, inplace=True)
    m1.columns = [prefix+str(col) for col in m1.columns]
    m1.to_csv(path)
