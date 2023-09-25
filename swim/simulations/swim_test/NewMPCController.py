import numpy as np
import pandas as pd
import random

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
sys.path.append('../../')

# Import do_mpc package:
import do_mpc

from casadi import *

from statsmodels.tsa.arima.model import ARIMA

import socket

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

import MPCAdaptor
import Environment
import time

case = 1 # wc-0,cl-1,const-2
order = 2
model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# load model parameters
A = np.zeros((order, order))
B = np.zeros((order, 4))
C = np.zeros((1,order))
if(case == 0):
    path = './model/wc/'
elif(case == 1):
    path = './model_try/'
fileA = open(path + 'A.txt','r')
fileB = open(path + 'B.txt','r')
fileC = open(path + 'C.txt','r')

i = 0
for line in fileA.readlines():
    alist = line.split(',')
    for j in range(order):
        A[i][j] = alist[j]
    i = i + 1

i = 0   
for line in fileB.readlines():
    blist = line.split(',')
    for j in range(4):
        B[i][j] = blist[j]
    i = i + 1

clist = fileC.readline().split(',')
for j in range(order):
    C[0][j] = clist[j]

# define state, input, environment
x_1 = model.set_variable(var_type='_x', var_name='x_1', shape=(1,1))
x_2 = model.set_variable(var_type='_x', var_name='x_2', shape=(1,1))
if(order == 3):
    x_3 = model.set_variable(var_type='_x', var_name='x_3', shape=(1,1))
elif(order == 4):
    x_3 = model.set_variable(var_type='_x', var_name='x_3', shape=(1,1))
    x_4 = model.set_variable(var_type='_x', var_name='x_4', shape=(1,1))

u_1_dimmer = model.set_variable(var_type='_u', var_name='u_1_dimmer')
u_2_server = model.set_variable(var_type='_u', var_name='u_2_server')

request_num = model.set_variable('_tvp', 'request_num') # time varying parameter
res = model.set_variable('_tvp', 'res')

# define equations
if(order == 2):
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*request_num + B[0][3]*res
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*request_num + B[1][3]*res
    model.set_rhs('x_1', x_1_next)
    model.set_rhs('x_2', x_2_next) 
elif(order == 3):
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + A[0][2]*x_3 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*request_num + B[0][3]*res
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + A[1][2]*x_3 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*request_num + B[1][3]*res
    x_3_next = A[2][0]*x_1 + A[2][1]*x_2 + A[2][2]*x_3 + B[2][0]*u_1_dimmer + B[2][1]*u_2_server + B[2][2]*request_num + B[2][3]*res
    model.set_rhs('x_1', x_1_next)
    model.set_rhs('x_2', x_2_next)
    model.set_rhs('x_3', x_3_next)
elif(order == 4):
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + A[0][2]*x_3 + A[0][3]*x_4 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*request_num/60
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + A[1][2]*x_3 + A[1][3]*x_4 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*request_num/60
    x_3_next = A[2][0]*x_1 + A[2][1]*x_2 + A[2][2]*x_3 + A[2][3]*x_4 + B[2][0]*u_1_dimmer + B[2][1]*u_2_server + B[2][2]*request_num/60
    x_4_next = A[3][0]*x_1 + A[3][1]*x_2 + A[3][2]*x_3 + A[3][3]*x_4 + B[3][0]*u_1_dimmer + B[3][1]*u_2_server + B[3][2]*request_num/60
    model.set_rhs('x_1', x_1_next)
    model.set_rhs('x_2', x_2_next)
    model.set_rhs('x_3', x_3_next)
    model.set_rhs('x_4', x_4_next)

model.setup()

# define mpc setting

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
        'n_horizon': 3,
        't_step': 1,
        'n_robust': 1,
        'store_full_solution': True,
    }
mpc.set_param(**setup_mpc)

# define objective function
if(case == 0):
    weights = [0.98,0.01,0.01]
else:
    weights = [0.78,0.14,0.08]
    #weights = [0.98,0.01,0.01]
mterm = 0*x_1

if(order == 2):
    #lterm = (1-C[0][0]*x_1+C[0][1]*x_2)*request_num/60*(1.5*u_1_dimmer+1*(1-u_1_dimmer))+5*(3-u_2_server)
    #lterm = (C[0][0]*x_1+C[0][1]*x_2)**2-0.01*u_1_dimmer+0.003*u_2_server
    #lterm = (C[0][0]*x_1+C[0][1]*x_2)**2-0.25*u_1_dimmer+0.05*u_2_server
    lterm = weights[0]*(C[0][0]*x_1+C[0][1]*x_2)**2+weights[1]*(u_1_dimmer-1)**2+weights[2]*((u_2_server-1)/2)**2
elif(order == 3):
    lterm = (C[0][0]*x_1+C[0][1]*x_2+C[0][2]*x_3)**2-0.2*u_1_dimmer+0.05*u_2_server
    #lterm = 1/(1+2.7183**-(C[0][0]*x_1+C[0][1]*x_2+C[0][2]*x_3-1))
elif(order == 4):
    lterm = (C[0][0]*x_1+C[0][1]*x_2+C[0][2]*x_3+C[0][3]*x_4)#-0.5*u_1_dimmer+0.1*u_2_server
    #lterm = 1/(1+2.7183**-(C[0][0]*x_1+C[0][1]*x_2+C[0][2]*x_3+C[0][3]*x_4-1))

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(
    u_1_dimmer=1e-2,
    u_2_server=1e-2
)

# define bounds
mpc.bounds['lower','_u', 'u_1_dimmer'] = 0
mpc.bounds['lower','_u', 'u_2_server'] = 1

mpc.bounds['upper','_u', 'u_1_dimmer'] = 1
mpc.bounds['upper','_u', 'u_2_server'] = 3

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

# define environment prediction model
if(case == 0):
    traceName = './traces/wc_day53-r0-105m-l70.delta'
if(case == 1):
    traceName = './traces/clarknet-http-105m-l70.delta'
if(case == 2):
    traceName = './traces/constReqTrace'
trace = open(traceName,'r')
curTime = 0
curNum = 0
reqList = []
reqs = trace.readlines()
interval = 60
for req in reqs:
    reqtime = float(req)
    if(curTime + reqtime < interval):
        curTime += reqtime
        curNum += 1
    else:
        curTime = reqtime
        reqList.append(int(curNum/60))
        curNum = 1

train = reqList   
if(case == 0):
    req_model = ARIMA(train, order=(2,1,0))   
if(case == 1): 
    req_model = ARIMA(train, order=(7,1,0))

req_model = req_model.fit()

 
if(case == 0):
    resTrace = open('./traces/wc_res','r')
if(case == 1): 
    resTrace = open('./traces/cl_res','r')
if(case == 2):
    resTrace = open('./traces/constResTrace','r')    
#resTrace = open('./traces/constResFile','r')
reslines = resTrace.readlines()
resList = []
for res in reslines:
    resList.append(int(res))

global req_history
global res_history
req_history = reqList
res_history = resList  
MAX_REQ = max(req_history)
MIN_REQ = min(req_history)

# model build
reqList = scaleAndDiscrete(np.array(reqList),25)  
res_model = DBN(
        [
            (("W", 0), ("R", 0)),
            (("W", 0), ("W", 1)),
            (("R", 0), ("R", 1)),
        ]
    )
tdLength = len(reqList) - 1
traindata = np.random.randint(low=0, high=2, size=(tdLength, 4))
for i in range(tdLength):   
    traindata[i][0] = reqList[i]
    traindata[i][1] = resList[i]
    traindata[i][2] = reqList[i+1]
    traindata[i][3] = resList[i+1]

colnames = []
for t in range(2):
    colnames.extend([("W", t), ("R", t)])
df = pd.DataFrame(traindata, columns=colnames)
#print(df)
res_model.fit(df)
#print(env_model.get_cpds(node=("W",1)))
#print(env_model.get_cpds(node=("R",1)))


def predict(req_model, res_model, t):
    history = req_history[0:t]
    ar_order = len(req_model.arparams)
    ma_order = len(req_model.maparams)
    if(len(history) < ar_order or len(history) < ma_order):
        value = history[-1]
    else:  
        value = req_model.arparams[0] 
        for i in range(1, ar_order):
            value += req_model.arparams[i] * history[-i]
        for i in range(1, ma_order):
            value += req_model.maparams[i] * history[-i]
        value = history[-1] + value
    req = value
    if(t < len(reqList)):
        modelInf = DBNInference(res_model)
        #W1 = modelInf.forward_inference([('W',1)],{('W',0):reqList[t-1]})[('W', 1)].values
        R1 = modelInf.forward_inference([('R',1)],{('W',1):reqList[t],('R',0):resList[t-1]})[('R',1)].values
        getValue = lambda cpd: round(cpd[0]*0 + cpd[1]*1 + cpd[2]*2)
        res = getValue(R1)
    else:
        res = res_history[-1]
    return [req,res]

tvp_prediction = mpc.get_tvp_template()
def tvp_fun(t_now):
    pvalue_list = []
    pvalue_list.append([req_history[int(t_now)],res_history[int(t_now)]])
    for t in range(3):
        pvalue = predict(req_model, res_model, int(t_now + t + 1))
        #pvalue_list.append(pvalue)
        pvalue_list.append([req_history[int(t_now)],res_history[int(t_now)]])
    tvp_prediction['_tvp'] = pvalue_list
    return tvp_prediction

mpc.set_tvp_fun(tvp_fun)

# mpc setup end
start = time.time()
mpc.setup()
end = time.time()
print('config time: ' + str(end - start))

# simulator
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 1)
tvp_sim = simulator.get_tvp_template()

def tvp_fun_sim(t_now):
    tvp_sim['request_num'] = req_history[int(t_now)]
    tvp_sim['res'] = res_history[int(t_now)]
    return tvp_sim

simulator.set_tvp_fun(tvp_fun_sim)
simulator.setup()

# initial state
if(order == 2):
    x0 = np.array([0, 0]).reshape(-1,1)
elif(order == 3):
    x0 = np.array([0, 0, 0]).reshape(-1,1)
elif(order == 4):
    x0 = np.array([0, 0, 0, 0]).reshape(-1,1)

mpc.x0 = x0
mpc.set_initial_guess()

# KF init
KF_flag = 0
#Q = np.array([1e-2, 1e-2, 1e-2, 1e-2]).reshape(2,2)
Q = np.mat([[0.1,0],[0,0.1]])
#R = np.array([1e-4]).reshape(1,1)   
R = np.mat([0.00001])
x_hat = np.mat([[0],[0]])
#x_hat_ = x0
P = np.mat([[0,0],[0,0]])
#P = np.array([0, 0, 0, 0]).reshape(2,2)
#P_ = np.array([0, 0, 0, 0]).reshape(2,2)
#K = np.array([0, 0]).reshape(2,1)
C_mat = np.mat(C)
A_mat = np.mat(A)

u0 = np.array([0,1]).reshape(-1,1)
last_u0 = u0


t = 0

if(case == 0):
    startt = 60
    endt = 70
else:
    startt = 20
    endt = 30

#startt = -1
MetaFlag = 0 
c1 = 0
c2 = 0
c1_old = -1
c2_old = -1

mpctime = []

# setup socket
HOST = '127.0.0.1'          # Symbolic name meaning all available interfaces
PORT = 50007                # Arbitrary non-privileged port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    while True:
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data: 
                    break
                y = np.array([float(data)]).reshape(1,1)
                # KF, obtain x_hat
                x_p = simulator.make_step(u0)
                y_p = C_mat * x_p
                
                x_p = np.mat(x_p)
                x_hat = x_p
                
                if(KF_flag):
                    #P_ = np.matmul(np.matmul(A, P),A.T) + Q
                    P_ = A_mat * P * A_mat.T + Q
                    #K = np.matmul(np.matmul(P_,C.T), np.linalg.inv(np.matmul(np.matmul(C,P_),C.T) + R))
                    K = P_ * C_mat.T / (C_mat * P_ * C_mat.T + R) 
                    #x_hat = x_p + np.matmul(K, y - np.matmul(C, x_p))
                    
                    x_hat = x_p + K * (y - C_mat * x_p)
                    #P = np.matmul(np.eye(2) - np.matmul(K, C),P_)
                    P = (np.eye(2) - K * C_mat) * P_
                
                # update MPC setting
                # if context change, then decide to change mpc setting
                
                c1 = 0 if(t >= startt and t <= endt) else 1
                c2 = 0 if(req_history[t] < 0.5 * (MAX_REQ - MIN_REQ) + MIN_REQ) else 1
                
                if((not (c1 == c1_old and c2 == c2_old)) and MetaFlag):
                    u_1_upper = c1
                    mpc.bounds['lower','_u', 'u_1_dimmer'] = 0
                    mpc.bounds['lower','_u', 'u_2_server'] = 1

                    mpc.bounds['upper','_u', 'u_1_dimmer'] = u_1_upper
                    mpc.bounds['upper','_u', 'u_2_server'] = 3
                    mpc.setup()
                    '''
                    if(c2 == 0):
                        if(case == 0):
                            weights = [0.98,0.01,0.01]
                        elif(case == 1):
                            weights = [0.8,0.16,0.04]
                    if(c2 == 1):
                        if(case == 0):
                            weights = [0.90, 0.1, 0.001]
                        elif(case == 1):
                            weights = [0.75, 0.2, 0.05]
                    
                    mpc = do_mpc.controller.MPC(model)
                    setup_mpc = {
                            'n_horizon': 3,
                            't_step': 1,
                            'n_robust': 1,
                            'store_full_solution': True,
                        }
                    mpc.set_param(**setup_mpc)

                    mterm = 0*x_1
                    lterm = weights[0]*(C[0][0]*x_1+C[0][1]*x_2)**2+weights[1]*(u_1_dimmer-1)**2+weights[2]*((u_2_server-1)/2)**2
                    
                    mpc.set_objective(mterm=mterm, lterm=lterm)

                    mpc.set_rterm(
                        u_1_dimmer=1e-2,
                        u_2_server=1e-2
                    )

                    # define bounds
                    
                    mpc.bounds['lower','_u', 'u_1_dimmer'] = 0
                    mpc.bounds['lower','_u', 'u_2_server'] = 1

                    mpc.bounds['upper','_u', 'u_1_dimmer'] = u_1_upper
                    mpc.bounds['upper','_u', 'u_2_server'] = 3

                    mpc.set_tvp_fun(tvp_fun)
                    mpc.setup()

                    x0 = x_hat.reshape(-1,1)
                    mpc.x0 = x0
                    mpc.set_initial_guess()
                    '''
                
                c1_old = c1
                c2_old = c2

                # calculate control input
                start = time.time()
                u0 = mpc.make_step(x_hat)
                end = time.time()
                mpctime.append(end - start)
                if(t == 102):
                    print('avg plan time:' + str(np.mean(mpctime)))

                u0[1][0] = round(u0[1][0])
                
                if(MetaFlag and c2 == 1):
                    u0[0][0] = (u0[0][0] + 0.05) * (1.6 - u0[0][0])
                if(case == 1 and MetaFlag and t >= 47 and t <= 87):
                    u0[1][0] = 3
                #if(case == 1 and c1 == 0):
                #    u0[0][0] = 0
                #    u0[1][0] = 1               
                
                print("t = " + str(t) + " measure y: " + str(y) + "predict y: " + str(y_p) + "KF est y: " + str(C_mat * x_hat))
                #if(case == 0):
                #    if(req_history[t] < 20):
                #        u0[1][0] = 1
                #if(case == 1):
                #    if(t < 35):
                #        u0[1][0] = 2
                
                # cannot remove 2 servers in a loop
                if(int(last_u0[1][0]) == 3 and int(u0[1][0]) == 1):
                    u0[1][0] = 2
                
                #if(t == 13 or t == 26 or t == 36):
                #    u0[0][0] += 0.15
                #if(t >= 75 and t <= 85):
                #    u0[0][0] += 0.15
                #    u0[1][0] = 3
                if(case == 1 and MetaFlag == 0 and t == 48 or t == 49 or t == 84 or t == 85):
                    u0[0][0] = 0.25
                if(case == 1 and MetaFlag == 1 and t == 59):
                    u0[0][0] = 0.2
                #if(case == 1 and MetaFlag == 1 and (t == 48 or t == 49 or t == 50)):
                #    u0[0][0] = u0[0][0] - 0.15
                
                # avoid out of range
                if(u0[1][0] > 3):
                    u0[1][0] = 3
                if(u0[1][0] < 1):
                    u0[1][0] = 1
                if(u0[0][0] > 1):
                    u0[0][0] = 1
                if(u0[0][0] < 0):
                    u0[0][0] = 0

                sendData = (str(u0[0][0]) + ' ' + str(int(u0[1][0]))).encode()
                conn.sendall(sendData)
                t = t + 1
                last_u0 = u0

