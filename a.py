# data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
color = sns.color_palette()

# machine learning
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

# keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.layer_utils import print_summary

# tqdm

np.random.seed(7)
tf.set_random_seed(777)  # reproducibility

# miscellaneous
import random
import os
import gc
import warnings
warnings.filterwarnings('ignore')
import operator
import json
from pprint import pprint
from xml.etree.ElementTree import parse
import datetime
import re
import math
import requests
from dateutil.relativedelta import relativedelta
import traceback

URL = 'http://api.hrfco.go.kr/220C13DE-9809-4646-9592-A60DC6C1ACB1/dam/list/1D/'
# utility functions


def getTimeStamp(fileName):
    startDate = fileName.split('.')[0][:4]
    endDate = fileName.split('.')[0][4:8]

    return datetime.datetime.strptime(startDate, timeformat)


dam_df = pd.read_json('data/damInfo.json', orient='records')
dam_df = dam_df.replace('', np.nan)
cols = ['dmobscd', 'obsnm', 'agcnm', 'addr',
        'etcaddr', 'lon', 'lat', 'pfh', 'fldlmtwl']
dam_df = dam_df[cols]
dam_df = dam_df.dropna(axis=0, how='any')

timeformat = '%y%m'
outformat = '%Y%m%d%H%M'
testYM = datetime.datetime.strptime('1607', timeformat)

targetYM = datetime.datetime.strptime('2001', timeformat)

testYM = testYM + relativedelta(months=12)

print(targetYM.strftime(outformat))

i = 0
while(True):
    if (testYM + relativedelta(months=12) < targetYM):
        resStr = testYM.strftime(
            outformat) + '/' + (testYM + relativedelta(months=12)).strftime(outformat)
        print(resStr)
        testYM += relativedelta(months=12)
    else:
        resStr = testYM.strftime(outformat) + '/' + \
            targetYM.strftime(outformat)
        print(resStr)
        break


def dam_data_load(damInfo, damName, startYM, endYM, columns=['dmobscd', 'ymdhm', 'swl', 'inf', 'sfw', 'ecpc', 'tototf', 'links']):
    timeformat = '%y%m'
    outformat = '%Y%m%d%H%M'
    res_df = pd.DataFrame(columns=columns)
    strURL = URL + damInfo[damInfo['obsnm'] ==
                           damName]['dmobscd'].values.astype(str)[0]

    startYM = datetime.datetime.strptime(startYM, timeformat)
    endYM = datetime.datetime.strptime(endYM, timeformat)

    while(True):
        if(startYM + relativedelta(months=12) < endYM):
            resURL = strURL + '/' + startYM.strftime(outformat) + '/' + (
                startYM + relativedelta(months=12)).strftime(outformat) + '.json'
            res = requests.get(resURL)
            file_df = pd.read_json(json.dumps(
                json.loads(res.text)['content']), orient='records')
            file_df = file_df.reindex(index=file_df.index[::-1])
            res_df = pd.concat([res_df, file_df]).reset_index(drop=True)
            startYM += relativedelta(months=12)
        else:
            resURL = strURL + '/' + \
                startYM.strftime(outformat) + '/' + \
                endYM.strftime(outformat) + '.json'
            res = requests.get(resURL)
            file_df = pd.read_json(json.dumps(
                json.loads(res.text)['content']), orient='records')
            file_df = file_df.reindex(index=file_df.index[::-1])
            res_df = pd.concat([res_df, file_df]).reset_index(drop=True)
            break

    res_df.drop_duplicates(subset=['ymdhm'], keep=False)

    return res_df


def dam_data_make(damInfo, damData, mergeCols=['dmobscd', 'pfh', 'fldlmtwl']):
    merge_df = pd.merge(damData, damInfo[mergeCols], on='dmobscd')

    # drop unnecessary columns
    merge_df = merge_df.drop(['dmobscd', 'sfw'], axis=1)

    # make new columns
    #merge_df['fldlmtwl'] = merge_df['fldlmtwl'].astype(float)
    #merge_df['isOverFLDLMTWL'] = merge_df.apply(lambda row : 1 if row.swl > row.fldlmtwl else 0, axis=1)

    # drop columns that already used
    merge_df = merge_df.drop(['pfh', 'fldlmtwl'], axis=1)

    # drop outliers
    merge_df = merge_df.drop(merge_df[merge_df['swl'] < 10].index, axis=0)

    #cols = ['ymdhm', 'swl', 'inf', 'ecpc', 'tototf', 'isOverFLDLMTWL']
    cols = ['swl', 'inf', 'ecpc', 'tototf']

    return merge_df[cols]


def build_dataset(damName, startYM, endYM):
    return dam_data_make(dam_df, dam_data_load(dam_df, damName, startYM, endYM))


def create_dataset(dataset, colNum, lookBack=1):
    dataX, dataY = [], []

    for i in range(len(dataset) - lookBack - 1):
        a = dataset[i:(i + 1)].values
        dataX.append(a)
        b = dataset[i + lookBack: i + lookBack + 1].values
        dataY.append(b)

    return np.array(dataX).reshape((-1, colNum)), np.array(dataY).reshape((-1, colNum))


def make_train_test(dataset, size, lookBack=1):
    train_size = int(len(dataset) * size)
    test_size = len(dataset) - train_size

    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, 4, lookBack)
    testX, testY = create_dataset(test, 4, lookBack)

    return trainX, trainY, testX, testY


def data_scaling_and_reshaping(trainX, trainY, testX, testY, scaler, lookBack):
    trainX = scaler.fit_transform(trainX)
    trainY, testX, testY = scaler.transform(
        trainY), scaler.transform(testX), scaler.transform(testY)

    return np.reshape(trainX, (-1, lookBack, trainX.shape[1])), trainY, np.reshape(testX, (-1, lookBack, testX.shape[1])), testY, scaler


def final_data_scaling_and_reshaping(Xdata, scaler, lookBack):
    Xdata = scaler.transform(Xdata)

    return np.reshape(Xdata, (-1, lookBack, Xdata.shape[1]))


def build_model(lookBack):
    model = Sequential()
    model.add(LSTM(16, input_shape=(lookBack, 4)))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def predict(model, trainX, testX, scaler):
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    return trainPredict, testPredict


def final_predict(model, Xdata, scaler):
    predict = model.predict(Xdata)
    predict = scaler.inverse_transform(predict)

    return predict[-2:]


def inverse_Ydata(trainY, testY, scaler):
    trainY = scaler.inverse_transform(trainY)
    testY = scaler.inverse_transform(testY)

    return trainY, testY


def print_score(trainPredict, trainY, testPredict, testY):
    trainScore = math.sqrt(mean_squared_error(
        trainY[:, 0], trainPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Test Score: %.2f RMSE' % (testScore))


def make_final_plot(dataset, trainPredict, testPredict, lookBack, target='swl', targetIndex=0):
    trainPredictPlot = np.empty_like(dataset[target])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[lookBack:len(
        trainPredict[:, targetIndex]) + lookBack] = trainPredict[:, targetIndex]

    testPredictPlot = np.empty_like(dataset[target])
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict[:, targetIndex]) + (lookBack * 2) +
                    1:len(dataset[target]) - 1] = testPredict[:, targetIndex]

    plt.subplots(1, 1, figsize=(16, 12))
    plt.plot(dataset[target])
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)

    plt.show()


def make_various_plot(trainPredict, trainY, testPredict, testY, targetIndex=0):
    trP = pd.Series(trainPredict[:, targetIndex])
    trY = pd.Series(trainY[:, targetIndex])
    teP = pd.Series(testPredict[:, targetIndex])
    teY = pd.Series(testY[:, targetIndex])

    f, axes = plt.subplots(2, 2, figsize=(16, 12))
    trP.plot(ax=axes[0, 0], color='red')
    trY.plot(ax=axes[0, 1], color='red')
    teP.plot(ax=axes[1, 0], color='green')
    teY.plot(ax=axes[1, 1], color='green')

    plt.show()


# params
lookBack = 1
startYM = '1601'
endYM = '1709'
resDict = {}
for code in dam_df['obsnm'].values:
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        timeformat = '%y%m'

        dataset = build_dataset(code, startYM, endYM)
        trainX, trainY, testX, testY = make_train_test(
            dataset, 0.98, lookBack=7)
        trainX, trainY, testX, testY, scaler = data_scaling_and_reshaping(
            trainX, trainY, testX, testY, scaler, 1)

        model = build_model(1)
        print('----------', code, 'training start----------')
        # model.summary()
        model.fit(trainX, trainY, nb_epoch=5, batch_size=1, verbose=1)

        resDict[code] = {}
        resDict[code]['model'] = model

        trainPredict, testPredict = predict(model, trainX, testX, scaler)
        trainY, testY = inverse_Ydata(trainY, testY, scaler)
        print_score(trainPredict, trainY, testPredict, testY)

        finalData = final_data_scaling_and_reshaping(dataset, scaler, 1)

        resDict[code]['final'] = final_predict(model, finalData, scaler)

        print(resDict[code]['final'])
        #make_final_plot(dataset, trainPredict, testPredict, 1)
        #make_various_plot(trainPredict, trainY, testPredict, testY)

        print('----------', code, 'training end----------')
    except Exception as error:
        print(traceback.format_exc())
        continue
