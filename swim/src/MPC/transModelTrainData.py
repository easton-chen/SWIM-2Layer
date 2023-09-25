import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    #print('brownout' + '\t' + 'serverNum' + '\t' + 'avgThroughput' + 'avgResponseTime' + '\t' 
        #+ 'basicMedianResponseTime' + '\t' + 'optMedianResponseTime' + '\t' 
        #+ 'timeoutRate')
    for i in range(len(brownoutSeries)):
        if(float(avgThroughputSeries[i]) != 0):
            avgThroughputSeries[i] = 1 / float(avgThroughputSeries[i]) 
            #print(brownoutSeries[i] + '\t' + serverNumSeries[i] + '\t' + str(avgThroughputSeries[i]) + '\t' + avgResponseTimeSeries[i] + '\t' 
            #    + basicMedianResponseTimeSeries[i] + '\t' + optMedianResponseTimeSeries[i]+ '\t' 
            #    + timeoutRateSeries[i])
            
            dataItem = []
            dataItem.append(1 - float(brownoutSeries[i])) # dimmer = 1 - brownout
            dataItem.append(serverNumSeries[i])
            dataItem.append(avgThroughputSeries[i])
            dataItem.append(resUtilSeries[i])
            #dataItem.append(avgResponseTimeSeries[i])
            '''
            if(float(avgResponseTimeSeries[i]) > 1):
                dataItem.append(1)
            else:
                dataItem.append(0)
            '''
            dataItem.append(timeoutRateSeries[i])
            dataList.append(dataItem)

    column = ['dimmer','serverNum','avgThroughput','resUtil','timeoutRate']
    dataDf = pd.DataFrame(columns=column,data=dataList)
    dataDf.to_csv('/home/czy/Desktop/SWIM-exp/swim/src/MPC/modelTrainData/trainDataRes7.csv')

def CobRA():
    dDimmerSeries = []
    dServerNumSeries = []
    dDimmerSeries.append(1 - float(brownoutSeries[0]))
    dServerNumSeries.append(serverNumSeries[0])
    for i in range(len(brownoutSeries)-1):
        dDimmerSeries.append(float(brownoutSeries[i]) - float(brownoutSeries[i+1]))
        dServerNumSeries.append(int(serverNumSeries[i+1]) - int(serverNumSeries[i]))

    for i in range(len(brownoutSeries)):
        if(float(avgThroughputSeries[i]) != 0):
            avgThroughputSeries[i] = 1 / float(avgThroughputSeries[i]) 
        dataItem = []
        # y: timeoutrate, dimmer, server; dU: dd,ds; a: req, res
        dataItem.append(timeoutRateSeries[i])
        dataItem.append(1 - float(brownoutSeries[i])) # dimmer = 1 - brownout
        dataItem.append(serverNumSeries[i])
        dataItem.append(dDimmerSeries[i])
        dataItem.append(dServerNumSeries[i])
        dataItem.append(avgThroughputSeries[i])
        dataItem.append(resUtilSeries[i])
        dataList.append(dataItem)

    column = ['timeoutRate','dimmer','serverNum','dDimmer','dServerNum','avgThroughput','resUtil']
    dataDf = pd.DataFrame(columns=column,data=dataList)
    dataDf.to_csv('/home/czy/Desktop/SWIM-exp/swim/src/MPC/modelTrainData/trainDataResCobRA1.csv')


if __name__ == "__main__": 
    # load data
    df = pd.read_csv('/home/czy/Desktop/SWIM-exp/results/SWIM_TRAIN/csv/Train-0.csv')
    df = pd.DataFrame(df, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
    brownout = df.loc[df['name'] == 'brownoutFactor:vector']
    serverNum = df.loc[df['name'] == 'activeServers:vector']
    avgResponseTime = df.loc[df['name'] == 'avgResponseTime:vector']
    avgThroughtput = df.loc[df['name'] == 'measuredInterarrivalAvg:vector']
    basicMedianResponseTime = df.loc[df['name'] == 'basicMedianResponseTime:vector']
    optMedianResponseTime = df.loc[df['name'] == 'optMedianResponseTime:vector']
    timeoutRate = df.loc[df['name'] == 'timeoutRate:vector']
    resUtil = df.loc[df['name'] == 'resUtil:vector']
    #print(avgResponseTime['vecvalue'].array[0])

    dataList = []
    avgResponseTimeSeries = avgResponseTime['vecvalue'].array[0].split(' ')
    brownoutSeries = brownout['vecvalue'].array[0].split(' ')
    serverNumSeries = serverNum['vecvalue'].array[0].split(' ')
    avgThroughputSeries = avgThroughtput['vecvalue'].array[0].split(' ')
    brownoutSeries = brownoutSeries[1:]
    serverNumSeries = serverNumSeries[1:]
    basicMedianResponseTimeSeries = basicMedianResponseTime['vecvalue'].array[0].split(' ')
    optMedianResponseTimeSeries = optMedianResponseTime['vecvalue'].array[0].split(' ')
    timeoutRateSeries = timeoutRate['vecvalue'].array[0].split(' ')
    resUtilSeries = resUtil['vecvalue'].array[0].split(' ')
    
    main()
    #CobRA()