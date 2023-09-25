import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv('./SWIM_SA/csv/Reactive-3.csv')
#df = pd.read_csv('./SWIM_TRAIN/csv/Train-0.csv')
df = pd.read_csv('./SWIM_TEST/csv/Test-0.csv')
#df = pd.read_csv('./all/new/CobRA-0.csv')
case = 0
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
#resUtilSeries = resUtil['vecvalue'].array[0].split(' ')

#print('brownout' + '\t' + 'serverNum' + '\t' + 'avgThroughput' + 'avgResponseTime' + '\t' 
    #+ 'basicMedianResponseTime' + '\t' + 'optMedianResponseTime' + '\t' 
#    + 'timeoutRate')

tlen = len(brownoutSeries)

dimmerSeries = []
accUtility = 0
accRevenue = 0
accPenalty = 0
accCost = 0
utilitySeries = []
qfCost = []
qfRevenue = []
qfTimeout = []
sat = []
if(case == 0):
    weights = [0.98,0.01,0.01]
else:
    weights = [0.80,0.16,0.04]

for i in range(tlen):
    avgThroughputSeries[i] = float(avgThroughputSeries[i]) 
    brownoutSeries[i] = float(brownoutSeries[i])
    dimmerSeries.append(1 - brownoutSeries[i])
    serverNumSeries[i] = float(serverNumSeries[i])
    timeoutRateSeries[i] = float(timeoutRateSeries[i])
    avgResponseTimeSeries[i] = float(avgResponseTimeSeries[i])
    #resUtilSeries[i] = float(resUtilSeries[i])
    
    if(avgThroughputSeries[i] != 0):
        avgThroughputSeries[i] = 1 / avgThroughputSeries[i]
        #print(brownoutSeries[i] + '\t' + serverNumSeries[i] + '\t' + str(avgThroughputSeries[i]) + '\t' + avgResponseTimeSeries[i] + '\t' 
        #    + basicMedianResponseTimeSeries[i] + '\t' + optMedianResponseTimeSeries[i]+ '\t' 
        #    + timeoutRateSeries[i])

        revenue = (1 - timeoutRateSeries[i]) * avgThroughputSeries[i] * (1.5 * (1 - brownoutSeries[i]) + 1 * brownoutSeries[i]) 
        penalty = 0.5 * timeoutRateSeries[i] * avgThroughputSeries[i]
        cost = 5 * (3 - serverNumSeries[i])
        accUtility = accUtility + revenue + cost - penalty  
        accRevenue = accRevenue + revenue
        accPenalty = accPenalty + penalty
        accCost = accCost + cost 
        utilitySeries.append(revenue + cost - penalty)
        
        

MAX_REQ = max(avgThroughputSeries)
for i in range(tlen): 
    # quality function for softgoal 
    qfco = 1 - 0.1 * serverNumSeries[i]
    qfCost.append(qfco)
    pire = (1 - timeoutRateSeries[i]) * (1.5 * (1 - brownoutSeries[i]) + 1 * brownoutSeries[i]) - 0.5 * timeoutRateSeries[i]
    if(pire > 1):
        qfre = 0.4 * pire + 0.4
    else:
        qfre = 0.533 * pire + 0.266
    qfRevenue.append(qfre)
    if(timeoutRateSeries[i] < 0.25):
        qfto = -2 * timeoutRateSeries[i] + 1
    else:
        qfto = -0.667 * timeoutRateSeries[i] + 0.667
    qfTimeout.append(qfto)
    if(avgThroughputSeries[i] < 3 / 4 * MAX_REQ):
        if(case == 0):
            weights = [0.98,0.01,0.01]
        elif(case == 1):
            weights = [0.8,0.16,0.04]
    else:
        if(case == 0):
            weights = [0.96, 0.03, 0.01]
        elif(case == 1):
            weights = [1, 0.3, 0.03]
    sat.append(weights[0] * qfto + weights[1] * qfre + weights[2] * qfco)
        

print("total utility = " + str(accUtility))     
print("total revenue = " + str(accRevenue))    
print("total penalty = " + str(accPenalty))    
print("total Cost = " + str(accCost))       

avgTimeoutRate = np.mean(timeoutRateSeries[2:])
avgRestime = np.mean(avgResponseTimeSeries[2:])
maxTimeoutRate = max(timeoutRateSeries[2:])
maxRestime = max(avgResponseTimeSeries[2:])
print("avg timeout:" + str(avgTimeoutRate) + " avg restime:" + str(avgRestime) + " max timeout: " + str(maxTimeoutRate) + " max restime:" + str(maxRestime))


resUtilSeries = []
if(case == 0):
    resFile = open("./wc_res")
    resUtils = resFile.readlines()
    for res in resUtils:
        resUtilSeries.append(float(res)/2)
if(case == 1):
    resFile = open("./cl_res")
    resUtils = resFile.readlines()
    for res in resUtils:
        resUtilSeries.append(float(res)/2)


fig,axarr = plt.subplots(6,1)  
fig.set_size_inches(6.4, 9.6)
plt.subplots_adjust(hspace=1)

for i in range(6):
    axarr[i].set_xlabel('t') 
    axarr[i].set_xlim(0,tlen) 

axarr[0].set_title('avgThroughput')
axarr[0].set_ylabel('avgThroughput')                          
axarr[0].set_ylim(0,max(avgThroughputSeries)) 
axarr[0].plot(range(tlen),avgThroughputSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[1].set_title('resUtil')
axarr[1].set_ylabel('resUtil')                          
axarr[1].set_ylim(0,3) 
axarr[1].plot(range(tlen),resUtilSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文

#axarr[1].set_title('avgresponsetime')
#axarr[1].set_ylabel('responsetime')                          
#axarr[1].set_ylim(0,1) 
#axarr[1].plot(range(tlen),avgResponseTimeSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文

axarr[2].set_title('dimmer')
axarr[2].set_ylabel('dimmer')                          
axarr[2].set_ylim(0,1) 
axarr[2].plot(range(tlen),dimmerSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[3].set_title('server')
axarr[3].set_ylabel('server')                          
axarr[3].set_ylim(0,3) 
axarr[3].plot(range(tlen),serverNumSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[4].set_title('timeoutRate')
axarr[4].set_ylabel('timeoutRate')                          
axarr[4].set_ylim(0,1) 
axarr[4].plot(range(tlen),timeoutRateSeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文本

axarr[5].set_title('Total Utility = ' + str(accUtility))
axarr[5].set_ylabel('Utility')                          
axarr[5].set_ylim(0,max(utilitySeries)) 
axarr[5].plot(range(tlen),utilitySeries,linestyle='--',alpha=0.5,color='r')   #线图：linestyle线性，alpha透明度，color颜色，label图例文

plt.show()


print("avg QFcost: " + str(np.mean(qfCost)) + "avg QFrevenue: " + str(np.mean(qfRevenue)) + "avg QFtimeout: " + str(np.mean(qfTimeout)))
print("Overall Satisfaction: " + str(np.mean(sat)))
x = np.arange(0,tlen)
fig0 = plt.figure(num=1, figsize=(8, 3)) 
ax0 = fig0.add_subplot(1,1,1)
ax0.plot(x, np.array(qfCost), label='Cost', linestyle=':')
ax0.plot(x, np.array(qfRevenue), label='Revenue', linestyle=':')
ax0.plot(x, np.array(qfTimeout), label='Timeout', linestyle=':')
ax0.legend(loc='best')
plt.show()
