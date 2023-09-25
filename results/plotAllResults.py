import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

case = 1
if(case == 0):
    dfRe = pd.read_csv('./all/new/Reactive-0.csv')
    dfCobra = pd.read_csv('./all/new/CobRA-0.csv')
    dfMy = pd.read_csv('./all/new/Ours-0.csv')
elif(case == 1):
    dfRe = pd.read_csv('./all/new/Reactive-1.csv')
    dfCobra = pd.read_csv('./all/new/CobRA-1.csv')
    dfMy = pd.read_csv('./all/new/Ours-1.csv')
dfRe = pd.DataFrame(dfRe, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
dfCobra = pd.DataFrame(dfCobra, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
dfMy = pd.DataFrame(dfMy, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])


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

def getData(df,flag):
    #df = pd.DataFrame(df, columns=['name','attrname','attrvalue','value','vectime','vecvalue'])
    brownout = df.loc[df['name'] == 'brownoutFactor:vector']
    serverNum = df.loc[df['name'] == 'activeServers:vector']
    avgResponseTime = df.loc[df['name'] == 'avgResponseTime:vector']
    avgThroughtput = df.loc[df['name'] == 'measuredInterarrivalAvg:vector']
    basicMedianResponseTime = df.loc[df['name'] == 'basicMedianResponseTime:vector']
    optMedianResponseTime = df.loc[df['name'] == 'optMedianResponseTime:vector']
    timeoutRate = df.loc[df['name'] == 'timeoutRate:vector']

    dataList = []
    avgResponseTimeSeries = avgResponseTime['vecvalue'].array[0].split(' ')
    dimmerSeries = brownout['vecvalue'].array[0].split(' ')
    serverNumSeries = serverNum['vecvalue'].array[0].split(' ')
    avgThroughputSeries = avgThroughtput['vecvalue'].array[0].split(' ')
    
    dimmerSeries = dimmerSeries[1:]
    serverNumSeries = serverNumSeries[1:]
    basicMedianResponseTimeSeries = basicMedianResponseTime['vecvalue'].array[0].split(' ')
    optMedianResponseTimeSeries = optMedianResponseTime['vecvalue'].array[0].split(' ')
    timeoutRateSeries = timeoutRate['vecvalue'].array[0].split(' ')

    qfCost = []
    qfRevenue = []
    qfTimeout = []
    sat = []

    

    tlen = len(dimmerSeries)

    accUtility = 0
    utilitySeries = []
    dDimmerSeries = []
    dServerNumSeries = []

    for i in range(tlen):
        avgThroughputSeries[i] = float(avgThroughputSeries[i]) 
        dimmerSeries[i] = 1 - float(dimmerSeries[i])    # change brownout value to dimmer value
        serverNumSeries[i] = float(serverNumSeries[i])
        timeoutRateSeries[i] = float(timeoutRateSeries[i])
        avgResponseTimeSeries[i] = float(avgResponseTimeSeries[i])

        if(i > 0 and i < tlen - 1):
            dDimmerSeries.append(abs(dimmerSeries[i] - 1 + float(dimmerSeries[i+1])))
            dServerNumSeries.append(abs(serverNumSeries[i] - float(serverNumSeries[i+1])))
        
        if(avgThroughputSeries[i] != 0):
            avgThroughputSeries[i] = 1 / avgThroughputSeries[i]

            

    #print("total utility = " + str(accUtility))  

    for i in range(tlen):

        revenue = (1 - timeoutRateSeries[i]) * avgThroughputSeries[i] * (1 * (1 - dimmerSeries[i]) + 1.5 * dimmerSeries[i]) - 0.5 * timeoutRateSeries[i] * avgThroughputSeries[i]
        cost = 5 * (3 - serverNumSeries[i])
        accUtility = accUtility + revenue + cost   
        utilitySeries.append(revenue + cost)
        # quality function for softgoal 
        qfco = 1 - 0.05 * serverNumSeries[i]
        qfCost.append(qfco)
        pire = (1.5 * dimmerSeries[i] + 1 * (1 - dimmerSeries[i]))
        qfre = -1.464 * pow(pire, 2) + 4.655 * pire - 2.691
        qfRevenue.append(qfre)
        if(timeoutRateSeries[i] < 0.2):
            qfto = -1.5 * timeoutRateSeries[i] + 1
        else:
            qfto = -0.875 * timeoutRateSeries[i] + 0.875
        qfTimeout.append(qfto)
        
        if(case == 0):
                weights = [0.33,0.33,0.33]
        elif(case == 1):
                weights = [0.33,0.33,0.33]
        
        sat.append(weights[0] * qfto + weights[1] * qfre + weights[2] * qfco)

    return sat, qfTimeout, qfRevenue, qfCost, accUtility, avgThroughputSeries, dimmerSeries, serverNumSeries, timeoutRateSeries, avgResponseTimeSeries, utilitySeries, dDimmerSeries, dServerNumSeries


def showStat(case, dDimmerSeries, dServerNumSeries, timeoutRateSeries, accUtility):
    dDimmerAvg = np.mean(dDimmerSeries)
    dServerNumAvg = np.mean(dServerNumSeries)
    timeoutRateSeries = timeoutRateSeries[5:]
    minY = min(timeoutRateSeries)
    maxY = max(timeoutRateSeries)
    devY = np.std(timeoutRateSeries)

    print(case + ' ' + str(dDimmerAvg) + ' ' + str(dServerNumAvg) + ' ' + str(minY) + ' ' + str(maxY)
        + ' ' + str(devY) + ' ' + str(accUtility))

#accUtilityRe, avgThroughputSeriesRe, dimmerSeriesRe, serverNumSeriesRe, timeoutRateSeriesRe, avgResponseTimeSeriesRe, utilitySeriesRe, dDimmerSeriesRe, dServerNumSeriesRe = getData(dfRe,0)
satBase, qfTimeoutBase, qfRevenueBase, qfCostBase, accUtilityCobra, avgThroughputSeriesCobra, dimmerSeriesCobra, serverNumSeriesCobra, timeoutRateSeriesCobra, avgResponseTimeSeriesCobra, utilitySeriesCobra, dDimmerSeriesCobra, dServerNumSeriesCobra = getData(dfCobra,1)
satMy, qfTimeoutMy, qfRevenueMy, qfCostMy, accUtilityMy, avgThroughputSeriesMy, dimmerSeriesMy, serverNumSeriesMy, timeoutRateSeriesMy, avgResponseTimeSeriesMy, utilitySeriesMy, dDimmerSeriesMy, dServerNumSeriesMy = getData(dfMy,2)



tlen = len(dimmerSeriesMy)
x = np.arange(tlen)

#plt.rcParams['figure.figsize']=(6.4, 12.8)
fig,axarr = plt.subplots(5,1)  
fig.set_size_inches(8.4, 12.8)
plt.subplots_adjust(hspace=1,right=0.75,top=0.95,bottom=0.05)

for i in range(5):
    axarr[i].set_xlim(0,tlen) 
axarr[4].set_xlabel('t') 

if(case == 0):
    axarr[0].set_title('request rate (WorldCup\'98)')
if(case == 1):
    axarr[0].set_title('request rate (ClarkNet)')
axarr[0].set_ylabel('request rate')                          
axarr[0].set_ylim(0,max(avgThroughputSeriesMy)) 
axarr[0].plot(range(tlen),avgThroughputSeriesMy,linestyle=':')    

axarr[1].set_title('resourse utilization')
axarr[1].set_ylabel('resourse utilization')                          
axarr[1].set_ylim(0,1.1) 
y_major_locator = MultipleLocator(0.5)
axarr[1].yaxis.set_major_locator(y_major_locator)
axarr[1].plot(range(tlen),resUtilSeries,linestyle=':')    

axarr[2].set_title('dimmer')
axarr[2].set_ylabel('dimmer')                          
axarr[2].set_ylim(0,1.1) 
y_major_locator = MultipleLocator(0.5)
axarr[2].yaxis.set_major_locator(y_major_locator)
#axarr[2].plot(range(tlen),dimmerSeriesRe,linestyle='--',alpha=0.5,color='r') 
axarr[2].plot(range(tlen),dimmerSeriesCobra) 
axarr[2].plot(range(tlen),dimmerSeriesMy)  
#axarr[2].spines['right'].set_visible(False)
#ax2 = axarr[2].twinx()
#ax2.plot(x, np.array(qfRevenueBase), linestyle=':', marker='*')
#ax2.plot(x, np.array(qfRevenueMy), linestyle=':', marker='*')


axarr[3].set_title('server num')
axarr[3].set_ylabel('server num')                          
axarr[3].set_ylim(0,3.1) 
y_major_locator = MultipleLocator(1)
axarr[3].yaxis.set_major_locator(y_major_locator)
#axarr[3].plot(range(tlen),serverNumSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[3].plot(range(tlen),serverNumSeriesCobra)
axarr[3].plot(range(tlen),serverNumSeriesMy)   

axarr[4].set_title('timeout rate')
axarr[4].set_ylabel('timeout rate')                          
axarr[4].set_ylim(0,1) 
#axarr[4].plot(range(tlen),timeoutRateSeriesRe,linestyle='--',alpha=0.5,color='r')
axarr[4].plot(range(tlen),timeoutRateSeriesCobra,linestyle='--')
axarr[4].plot(range(tlen),timeoutRateSeriesMy,linestyle='--')   

#axarr[5].set_title('Utility')
#axarr[5].set_ylabel('Utility')                          
#axarr[5].set_ylim(0,max(utilitySeriesMy)) 
#axarr[5].plot(range(tlen),utilitySeriesRe,linestyle='--',alpha=0.5,color='r') 
#axarr[5].plot(range(tlen),utilitySeriesCobra,linestyle='--',alpha=0.5,color='b')
#saxarr[5].plot(range(tlen),utilitySeriesMy,linestyle='--',alpha=0.5,color='o')  

axarr[2].legend(labels=['Base MPC','Context-aware MPC'],loc=(1.02,0.6))
plt.show()

#showStat('Reactive', dDimmerSeriesRe, dServerNumSeriesRe, timeoutRateSeriesRe, accUtilityRe)
showStat('CobRA', dDimmerSeriesCobra, dServerNumSeriesCobra, timeoutRateSeriesCobra, accUtilityCobra)
showStat('Ours', dDimmerSeriesMy, dServerNumSeriesMy, timeoutRateSeriesMy, accUtilityMy)


x = np.arange(0,tlen)
fig0 = plt.figure(num=1, figsize=(8, 8)) 
plt.subplots_adjust(hspace=1,right=0.75,top=0.95,bottom=0.05)
ax0 = fig0.add_subplot(4,1,1)
ax0.set_title('Low Cost')
ax0.set_ylim(0.4,1.1) 
y_major_locator = MultipleLocator(0.2)
ax0.plot(x, np.array(qfCostBase), linestyle=':', marker=',')
ax0.plot(x, np.array(qfCostMy), linestyle=':', marker=',')
#ax0.legend(labels=['Base','Context-aware'], loc='best')

#fig1 = plt.figure(num=1, figsize=(8, 4)) 
ax1 = fig0.add_subplot(4,1,2)
ax1.set_title('High Revenue')
ax1.set_ylim(0.4,1.1) 
y_major_locator = MultipleLocator(0.2)
ax1.plot(x, np.array(qfRevenueBase), linestyle=':', marker=',')
ax1.plot(x, np.array(qfRevenueMy), linestyle=':', marker=',')
#ax1.legend(labels=['Base','Context-aware'], loc='best')

#fig2 = plt.figure(num=1, figsize=(8, 4)) 
ax2 = fig0.add_subplot(4,1,3)
ax2.set_title('Low Timeout Rate')
ax2.set_ylim(0.4,1.1) 
y_major_locator = MultipleLocator(0.2)
ax2.plot(x, np.array(qfTimeoutBase), linestyle=':', marker=',')
ax2.plot(x, np.array(qfTimeoutMy), linestyle=':', marker=',')
ax2.legend(labels=['Base MPC','Context-aware MPC'], loc=(1.02,1.0))

#fig3 = plt.figure(num=1, figsize=(8, 4)) 
ax3 = fig0.add_subplot(4,1,4)
ax3.set_title('Overall Satisfaction')
ax3.set_ylim(0.4,1.1) 
y_major_locator = MultipleLocator(0.2)
ax3.plot(x, np.array(satBase), linestyle=':', marker=',')
ax3.plot(x, np.array(satMy), linestyle=':', marker=',')
#ax3.legend(labels=['Base','Context-aware'], loc='best')
plt.show()

print('ovreall satisfaction of base mpc:' + str(np.mean(satBase)))
print('ovreall satisfaction of context-aware mpc:' + str(np.mean(satMy)))
