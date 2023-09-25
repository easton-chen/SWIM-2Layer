from cProfile import label
#from importlib.metadata import MetadataPathFinder
from ipaddress import collapse_addresses

import pgmpy.models
import pgmpy.inference
import numpy as np
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import random

# The main entry point for this module
def example1():
    # Create a dynamic bayesian network
    model = pgmpy.models.DynamicBayesianNetwork()
    # Add nodes
    model.add_nodes_from(['Weather', 'Umbrella'])
    # Print nodes
    print('--- Nodes ---')
    print(model.nodes())
    # Add edges
    model.add_edges_from([(('Umbrella',0), ('Weather',0)),
                          (('Weather',0), ('Weather',1)),
                          (('Umbrella',0), ('Umbrella',1))])
    # Print edges
    print('--- Edges ---')
    print(model.edges())
    print()
    # Print parents
    print('--- Parents ---')
    print('Umbrella 0: {0}'.format(model.get_parents(('Umbrella', 0))))
    print('Weather 0: {0}'.format(model.get_parents(('Weather', 0))))
    print('Weather 1: {0}'.format(model.get_parents(('Weather', 1))))
    print('Umbrella 1: {0}'.format(model.get_parents(('Umbrella', 1))))
    print()

    # Add probabilities
    weather_cpd = pgmpy.factors.discrete.TabularCPD(('Weather', 0), 2, [[0.1, 0.8], 
                                                                        [0.9, 0.2]], 
                                                       evidence=[('Umbrella', 0)], 
                                                       evidence_card=[2])
    umbrella_cpd = pgmpy.factors.discrete.TabularCPD(('Umbrella', 1), 2, [[0.5, 0.5], 
                                                                          [0.5, 0.5]], 
                                                     evidence=[('Umbrella', 0)], 
                                                     evidence_card=[2])
    transition_cpd = pgmpy.factors.discrete.TabularCPD(('Weather', 1), 2, [[0.25, 0.9, 0.1, 0.25], 
                                                                           [0.75, 0.1, 0.9, 0.75]], 
                                                   evidence=[('Weather', 0), ('Umbrella', 1)], 
                                                   evidence_card=[2, 2])

    # Add conditional probability distributions (cpd:s)
    model.add_cpds(weather_cpd, umbrella_cpd, transition_cpd)
    # This method will automatically re-adjust the cpds and the edges added to the bayesian network.
    model.initialize_initial_state()
    # Check if the model is valid, throw an exception otherwise
    model.check_model()

    # Print probability distributions
    print('Probability distribution, P(Weather(0) | Umbrella(0)')
    print(weather_cpd)
    print()
    print('Probability distribution, P(Umbrella(1) | Umbrella(0)')
    print(umbrella_cpd)
    print()
    print('Probability distribution, P(Weather(1) | Umbrella(1), Weather(0)')
    print(transition_cpd)
    print()

    # Make inference
    map = {0: 'Sunny', 1: 'Rainy' }
    dbn_inf = pgmpy.inference.DBNInference(model)
    
    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):0, ('Weather', 0):0})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : Yes, Weather(0): Sunny): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()

    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):0, ('Weather', 0):1})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : Yes, Weather(0): Rainy): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()
    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):1, ('Weather', 0):0})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : No, Weather(0): Sunny): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()
    result = dbn_inf.forward_inference([('Weather', 1)], {('Umbrella', 1):1, ('Weather', 0):1})
    arr = result[('Weather', 1)].values
    print()
    print('Prediction (Umbrella(1) : No, Weather(0): Rainy): {0} ({1} %)'.format(map[np.argmax(arr)], np.max(arr) * 100))
    print()


def example2():
    model = DBN(
        [
            (("W", 0), ("R", 0)),
            (("W", 0), ("W", 1)),
            (("R", 0), ("R", 1)),
        ]
    )
    data = np.random.randint(low=0, high=2, size=(10, 6))
    #print(data)
    colnames = []
    for t in range(3):
        colnames.extend([("W", t), ("R", t)])
    df = pd.DataFrame(data, columns=colnames)
    #print(df)
    model.fit(df)
    print(model.get_cpds(node=("R",1)))

def scaleAndDiscrete(data,flag=0):
    if(flag == 0):
        minValue = data.min()
        maxValue = data.max()
        for i in range(len(data)):
            data[i] = (data[i] - minValue) / (maxValue - minValue) * 100
            #print(data[i])
            data[i] = int(data[i] / 34)
    else:
        for i in range(len(data)):
            data[i] = int(data[i] / flag)
    return data

def main():
    data = pd.read_csv('/home/czy/Desktop/SWIM-exp/swim/src/MPC/DBN/PerfResult.csv',usecols=['Value','Timestamp','MetricId','Entity'])
    #print(data.head())
    data = data[data['Entity']=='php5_server']
    #data = data[data['Entity']=='php7_server']
    #data = data[data['Entity']=='media_server']
    CpuPercentData = np.array(data[data['MetricId']=='cpu.usage.average']['Value'])
    CpuHzData = np.array(data[data['MetricId']=='cpu.usagemhz.average']['Value'])
    MemData = np.array(data[data['MetricId']=='mem.usage.average']['Value'])
    DiskData = np.array(data[data['MetricId']=='disk.usage.average']['Value'])
    NetData = np.array(data[data['MetricId']=='net.usage.average']['Value'])

    #scale(CpuHzData)
    #print(CpuHzData)
    
    #dataMat = np.array([CpuPercentData,CpuHzData,MemData,DiskData,NetData])
    #corr = np.corrcoef(dataMat)
    #print(corr) 
    
    scaleAndDiscrete(NetData)
    scaleAndDiscrete(CpuPercentData)
    #x = range(len(CpuPercentData))
    #plt.plot(x,NetData,label='net')
    #plt.plot(x,CpuPercentData,label='cpu percent')
    #plt.plot(x,scaleAndDiscrete(CpuHzData),label='cpu hz')
    #plt.plot(x,scaleAndDiscrete(MemData),label='mem')
    #plt.plot(x,scaleAndDiscrete(DiskData),label='disk')
    #plt.legend()
    #plt.show()

    model = modelBuild(NetData,CpuPercentData)

    #trace = open('/home/czy/Desktop/SWIM-exp/swim/src/MPC/traces/wc_day53-r0-105m-l70.delta','r')
    trace = open('/home/czy/Desktop/SWIM-exp/swim/src/MPC/traces/clarknet-http-105m-l70.delta','r')
    curTime = 0
    curTime = 0
    curNum = 0
    reqList = []
    reqs = trace.readlines()
    for req in reqs:
        time = float(req)
        if(curTime + time < 60):
            curTime += time
            curNum += 1
        else:
            curTime = time
            reqList.append(int(curNum / 60))
            curNum = 1

    reqList = scaleAndDiscrete(np.array(reqList),25)
    #for i in range(len(reqList)):
    #    print(str(i) + ':' + str(reqList[i]))

    modelInf = DBNInference(model)
    resUtil = []
    resUtil.append(0)
    reqPredict = []
    #W1 = modelInf.forward_inference([('W',1)],{('W',0):1})[('W', 1)].values
    getValue = lambda cpd: round(cpd[0]*0 + cpd[1]*1 + cpd[2]*2)
    #print(getValue(W1))
    
    for i in range(len(reqList) - 1):
        #W1 = modelInf.forward_inference([('W',1)],{('W',0):reqList[i]})[('W', 1)].values
        R1 = modelInf.forward_inference([('R',1)],{('W',1):reqList[i+1],('R',0):resUtil[i]})[('R',1)].values
        #reqPredict.append(getValue(W1))
        ran = np.random.random()
        if(ran < R1[0]):
            resUtil.append(0)
        elif(ran < (R1[0]+R1[1])):
            resUtil.append(1)
        else:
            resUtil.append(2)
        #resUtil.append(getValue(R1))
    
    resTraceName = '/home/czy/Desktop/SWIM-exp/swim/src/MPC/traces/cl_res'
    resTrace = open(resTraceName,'w')
    for i in range(len(resUtil)):
        resTrace.write(str(resUtil[i]) + '\n')

    for i in range(len(resUtil)):
        print(str(reqList[i]) + ' ' + str(resUtil[i]))
        #print(resUtil[i])
    
    model1 = modelBuild(reqList, resUtil)

def modelBuild(data1,data2):
    model = DBN(
        [
            (("W", 0), ("R", 0)),
            (("W", 0), ("W", 1)),
            (("R", 0), ("R", 1)),
        ]
    )
    
    # generate dbn training data
    tdLength = len(data1) - 1
    traindata = np.random.randint(low=0, high=2, size=(tdLength, 4))
    for i in range(tdLength):
        traindata[i][0] = data1[i]
        traindata[i][1] = data2[i]
        traindata[i][2] = data1[i+1]
        traindata[i][3] = data2[i+1]


    colnames = []
    for t in range(2):
        colnames.extend([("W", t), ("R", t)])
    df = pd.DataFrame(traindata, columns=colnames)
    #print(df)
    model.fit(df)
    print(model.get_cpds(node=("W",1)))
    print(model.get_cpds(node=("R",1)))

    return model




def generateResData():
    len = 210    
    dataFile = open('/home/czy/Desktop/SWIM-exp/swim/simulations/swim/traces/sourceFile','w')
    for i in range(len):
        res = random.randint(0,2) 
        dataFile.write(str(res) + '\n')
    dataFile.close()
    
if __name__ == "__main__": 
    main()
    #generateResData()
    
    
    