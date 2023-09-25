import numpy as np

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
sys.path.append('../../')

# Import do_mpc package:
import do_mpc

from casadi import *

from statsmodels.tsa.arima.model import ARIMA

import socket

case = 0 # wc-0,cl-1
order = 2
model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# load model parameters
A = np.zeros((order, order))
B = np.zeros((order, 3))
C = np.zeros((1,order))
path = './model_old/'
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
    for j in range(3):
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

# define equations
if(order == 2):
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*request_num/60
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*request_num/60
    model.set_rhs('x_1', x_1_next)
    model.set_rhs('x_2', x_2_next) 
elif(order == 3):
    x_1_next = A[0][0]*x_1 + A[0][1]*x_2 + A[0][2]*x_3 + B[0][0]*u_1_dimmer + B[0][1]*u_2_server + B[0][2]*request_num/60
    x_2_next = A[1][0]*x_1 + A[1][1]*x_2 + A[1][2]*x_3 + B[1][0]*u_1_dimmer + B[1][1]*u_2_server + B[1][2]*request_num/60
    x_3_next = A[2][0]*x_1 + A[2][1]*x_2 + A[2][2]*x_3 + B[2][0]*u_1_dimmer + B[2][1]*u_2_server + B[2][2]*request_num/60
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
mterm = 0*x_1

if(order == 2):
    #lterm = (1-C[0][0]*x_1+C[0][1]*x_2)*request_num/60*(1.5*u_1_dimmer+1*(1-u_1_dimmer))+5*(3-u_2_server)
    if(case == 0):
        lterm = (C[0][0]*x_1+C[0][1]*x_2)**2-0.2*u_1_dimmer+0.002*u_2_server
    if(case == 1):
        lterm = (C[0][0]*x_1+C[0][1]*x_2)**2-0.14*u_1_dimmer+0.001*u_2_server
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

# define environment prediction model
tvp_prediction = mpc.get_tvp_template()

if(case == 0):
    traceName = './traces/wc_day53-r0-105m-l70.delta'
if(case == 1):
    traceName = './traces/clarknet-http-105m-l70.delta'
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
        reqList.append(curNum)
        curNum = 1

train = reqList   
if(case == 0):
    env_model = ARIMA(train, order=(2,1,0))   
if(case == 1): 
    env_model = ARIMA(train, order=(7,1,0))
global env_model_fit
env_model_fit = env_model.fit()

global history
history = reqList

def predict(model, history_, t):
    history = history_[0:t]
    ar_order = len(model.arparams)
    ma_order = len(model.maparams)
    if(len(history) < ar_order or len(history) < ma_order):
        value = history[-1]
    else:  
        value = model.arparams[0] 
        for i in range(1, ar_order):
            value += model.arparams[i] * history[-i]
        for i in range(1, ma_order):
            value += model.maparams[i] * history[-i]
        value = history[-1] + value
    return value

def tvp_fun(t_now):
    pvalue_list = []
    pvalue_list.append(history[int(t_now)])
    for t in range(3):
        pvalue = predict(env_model_fit, history, int(t_now + t + 1))
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
    tvp_sim['request_num'] = history[int(t_now)]
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
R = np.array([1e-5]).reshape(1,1)   
x_hat = x0
#x_hat_ = x0
P = np.array([0, 0, 0, 0]).reshape(2,2)
#P_ = np.array([0, 0, 0, 0]).reshape(2,2)
K = np.array([0, 0]).reshape(2,1)
u0 = np.array([0,1]).reshape(-1,1)

t = 0

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
                P = np.matmul(np.matmul(A, P),A.T) + Q
                K = np.matmul(np.matmul(P,C.T), np.linalg.inv(np.matmul(np.matmul(C,P),C.T) + R))
                x_hat = x_p + np.matmul(K, y - np.matmul(C, x_p))
                P = np.matmul(np.eye(2) - np.matmul(K, C),P)
                u0 = mpc.make_step(x_hat)
                if(case == 0):
                    if(reqList[t] / 60 < 20):
                        u0[1][0] = 1
                #if(case == 1):
                #    if(reqList[t] / 60 < 15):
                #        u0[1][0] = 1
                sendData = (str(u0[0][0]) + ' ' + str(int(u0[1][0]))).encode()
                conn.sendall(sendData)
                t = t + 1