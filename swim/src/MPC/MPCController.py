from glob import glob
import numpy as np
import pandas as pd

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
sys.path.append('../../')

# Import do_mpc package:
import do_mpc

from casadi import *

from statsmodels.tsa.arima.model import ARIMA

import socket

import pgmpy.models
import pgmpy.inference

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

order = 2
model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# load model parameters
A = np.zeros((order, order))
B = np.zeros((order, 4))
C = np.zeros((1,order))
path = './'
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
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*(request_num+1)*25 + B[0][3]*res
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*(request_num+1)*25 + B[1][3]*res
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
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + A[0][2]*x_3 + A[0][3]*x_4 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*request_num + B[0][3]*res
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + A[1][2]*x_3 + A[1][3]*x_4 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*request_num + B[1][3]*res
    x_3_next = A[2][0]*x_1 + A[2][1]*x_2 + A[2][2]*x_3 + A[2][3]*x_4 + B[2][0]*u_1_dimmer + B[2][1]*u_2_server + B[2][2]*request_num + B[2][3]*res
    x_4_next = A[3][0]*x_1 + A[3][1]*x_2 + A[3][2]*x_3 + A[3][3]*x_4 + B[3][0]*u_1_dimmer + B[3][1]*u_2_server + B[3][2]*request_num + B[3][3]*res
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
mterm = 0*x_1

if(order == 2):
    lterm = (C[0][0]*x_1+C[0][1]*x_2)-0.2*u_1_dimmer+0.05*u_2_server
    #lterm = 1/(1+2.7183**-(C[0][0]*x_1+C[0][1]*x_2-1))
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

traceName = './traces/wc_day53-r0-105m-l70.delta'
trace = open(traceName,'r')
curTime = 0
curNum = 0
reqList = []
reqs = trace.readlines()
interval = 60
for req in reqs:
    time = float(req)
    if(curTime + time < interval):
        curTime += time
        curNum += 1
    else:
        curTime = time
        reqList.append(int(curNum / 60))
        curNum = 1

reqList = scaleAndDiscrete(np.array(reqList),25)      

resTrace = open('./traces/wc_res','r')
reslines = resTrace.readlines()
resList = []
for res in reslines:
    resList.append(int(res))

global req_history
global res_history
req_history = reqList
res_history = resList

# model build
#global env_model
env_model = DBN(
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
env_model.fit(df)
print(env_model.get_cpds(node=("W",1)))
print(env_model.get_cpds(node=("R",1)))

def predict(model, t):
    modelInf = DBNInference(model)
    W1 = modelInf.forward_inference([('W',1)],{('W',0):reqList[t-1]})[('W', 1)].values
    R1 = modelInf.forward_inference([('R',1)],{('W',1):reqList[t],('R',0):resList[t-1]})[('R',1)].values
    getValue = lambda cpd: round(cpd[0]*0 + cpd[1]*1 + cpd[2]*2)
    req = getValue(W1)
    res = getValue(R1)
    return [req,res]


# define environment prediction model
tvp_prediction = mpc.get_tvp_template()

def tvp_fun(t_now):
    pvalue_list = []
    pvalue_list.append([req_history[int(t_now)],res_history[int(t_now)]])
    for t in range(3):
        pvalue = predict(env_model, int(t_now + t + 1))
        pvalue_list.append(pvalue)
    tvp_prediction['_tvp'] = pvalue_list
    return tvp_prediction


mpc.set_tvp_fun(tvp_fun)

# mpc setup end
mpc.setup()

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
Q = np.array([1e-5, 1e-5, 1e-5, 1e-5]).reshape(2,2)
R = 1e-5    
x_hat = x0
#x_hat_ = x0
P = np.array([0, 0, 0, 0]).reshape(2,2)
#P_ = np.array([0, 0, 0, 0]).reshape(2,2)
K = np.array([0, 0]).reshape(2,1)
u0 = np.array([1,1]).reshape(-1,1)

# setup socket
HOST = '127.0.0.1'          # Symbolic name meaning all available interfaces
PORT = 50007                # Arbitrary non-privileged port
print('Start connecting...')
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data: 
                break
            y = float(data)
            # KF, obtain x_hat
            x_p = simulator.make_step(u0)
            P = np.matmul(np.matmul(A, P),A.T) + Q
            K = np.matmul(np.matmul(P,C.T), np.linalg.inv(np.matmul(np.matmul(C,P),C.T) + R))
            x_hat = x_p + np.matmul(K, y - np.matmul(C, x_p))
            P = np.matmul(np.eye(2) - np.matmul(K, C),P)
            u0 = mpc.make_step(x_hat)
            sendData = (str(u0[0][0]) + str(u0[1][0])).encode()
            conn.sendall(str(u0))
