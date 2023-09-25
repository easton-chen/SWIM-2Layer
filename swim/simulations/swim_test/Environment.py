from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

class Environment:
    def __init__(self, case):
        #self.requestrate = 0
        #self.resutil = 0
        if(case == 0):
            traceName = './traces/wc_day53-r0-105m-l70.delta'
            resTrace = open('./traces/wc_res','r')
        if(case == 1):
            traceName = './traces/clarknet-http-105m-l70.delta'
            resTrace = open('./traces/cl_res','r')
        trace = open(traceName,'r')
        curTime = 0
        curNum = 0
        self.reqList = []
        reqs = trace.readlines()
        interval = 60
        for req in reqs:
            time = float(req)
            if(curTime + time < interval):
                curTime += time
                curNum += 1
            else:
                curTime = time
                self.reqList.append(int(curNum/60))
                curNum = 1
        train = self.reqList   
        if(case == 0):
            self.req_model = ARIMA(train, order=(2,1,0))   
        if(case == 1): 
            self.req_model = ARIMA(train, order=(7,1,0))

        self.req_model = self.req_model.fit()

        reslines = resTrace.readlines()
        self.resList = []
        for res in reslines:
            self.resList.append(int(res))
        train = self.resList
        if(case == 0):
            self.res_model = ARIMA(train, order=(4,1,0))   
        if(case == 1): 
            self.res_model = ARIMA(train, order=(1,1,0))
        self.res_model = self.res_model.fit()

    def update(self, t):
        pass

    def predict(self, t):
        history = self.reqList[0:t]
        ar_order = len(self.req_model.arparams)
        ma_order = len(self.req_model.maparams)
        if(len(history) < ar_order or len(history) < ma_order):
            value = history[-1]
        else:  
            value = self.req_model.arparams[0] 
            for i in range(1, ar_order):
                value += self.req_model.arparams[i] * history[-i]
            for i in range(1, ma_order):
                value += self.req_model.maparams[i] * history[-i]
            value = history[-1] + value
        req = value
        history = self.resList[0:t]
        ar_order = len(self.res_model.arparams)
        ma_order = len(self.res_model.maparams)
        if(len(history) < ar_order or len(history) < ma_order):
            value = history[-1]
        else:  
            value = self.res_model.arparams[0] 
            for i in range(1, ar_order):
                value += self.res_model.arparams[i] * history[-i]
            for i in range(1, ma_order):
                value += self.res_model.maparams[i] * history[-i]
            value = history[-1] + value
        res = value
        return [req,res]
        #return self.reqList[t], self.resList[t]

    def test(self):
        size = int(len(self.resList) * 0.5)
        testList = self.resList[size:len(self.resList)]
        prediction = []
        num = 0
        for t in range(len(testList)):
            yhat = self.predict(size + t)
            prediction.append(yhat[1])
            if(testList[t] == round(yhat[1])):
                num = num + 1
            print("prediction = %d, true value = %d" % (round(yhat[1]),testList[t]))

        #error = mean_squared_error(testList,prediction)
        #RMSE = math.sqrt(error)
        #print("test RMSE: %.3f" % RMSE)
        accuracy = 1.0 * num / len(testList)
        print(accuracy)



env = Environment(1)
    