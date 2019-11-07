#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from numpy import linalg as LA
import sympy
import matplotlib.pyplot as plt
import random
import math


# # E.1 Python exercise
# 
# Write a Python script to implement the backpropagation algorithm for a 1 − S
# 1 − 1 network.
# 
# Write the program using matrix operations, as in Eq. (11.41) to Eq. (11.47). Choose the initial weights  and biases to be random numbers uniformly distributed between -0.5 and 0.5 (using the function rand), and train the network to approximate the function g(p) = e−abs(p) ×sin(π p) for −2 ≤ p ≤ 2
# Use S1 = 2 and S1 = 10. Experiment with several different values for the learning rate α , and use several different initial conditions. Discuss the convergence properties of the algorithm as the learning rate changes.
#
# 
# Note: You are not allowed to use any ML packages, you are allowed to use basic python packages (numpy and etc). Please code the summary of backdrop equations (that is the onlything you need).
# 
# Bonus points:
# Write your code in a format that you can enter any number of neurons in hidden layer.
# 

#S1=2
S1=10

W1_0=np.zeros((S1,1))
W2_0=np.zeros((1,S1))
b1_0=np.zeros((S1,1))
b2_0=np.array(np.random.uniform(low=-0.5, high=0.5))

for i in range(0,S1):
    W1_0[i]=np.random.uniform(low=-0.5,high=0.5)
    W2_0[0][i]=np.random.uniform(low=-0.5,high=0.5)
    b1_0[i] = np.random.uniform(low=-0.5, high=0.5)


# W1_0=np.array([[np.random.uniform(low=-0.5,high=0.5)],[np.random.uniform(low=-0.5,high=0.5)]])
# b1_0=np.array([[np.random.uniform(low=-0.5,high=0.5)],[np.random.uniform(low=-0.5,high=0.5)]])
# W2_0=np.array([[np.random.uniform(low=-0.5,high=0.5)],[np.random.uniform(low=-0.5,high=0.5)]]).T
#b2_0=np.array((np.random.uniform(low=-0.5,high=0.5)))

alpha=.01
p=np.linspace(-2,2,200)
epochs=15000


def g(n):
    # for −2 ≤ p ≤ 2
    gn=math.exp(-np.absolute(n))*np.sin(np.pi*n)
    return gn

t=[]
for i in p:
    t.append(g(i))
print(t)


def logsig(n):
    a=1+np.exp(-n)
    r= 1/(a)
    return r
#
# def stochnetworkfunc(p,W1_0,b1_0,W2_0,b2_0,alpha):
#     mse=np.zeros((epochs))
#     for j in range(0,epochs):
#         e1 = np.zeros(( len(p)))
#         for i in range(len(p)):
#             n=np.dot(W1_0, p[i]) + b1_0
#             a1_1 =logsig(n)
#             a2_1=((np.dot(W2_0, a1_1)) + b2_0)
#             e1[i] = g(p[i]) - a2_1
#             s2=(-2)*1*e1[i]
#             s1=np.dot(np.diag(((1-a1_1)*a1_1).flatten()),(W2_0).T)*s2
#             W2_0 = W2_0 - (alpha*np.dot(s2,a1_1.T))
#             W1_0 = W1_0 - (alpha*s1*p[i])
#             b2_0 = b2_0 - alpha*s2
#             b1_0= b1_0-alpha*s1
#
#         mse[j]=np.dot(e1.T,e1)
#     return np.array(W1_0), np.array(b1_0),np.array(W2_0), np.array(b2_0),e1,mse
#
# #stochnetworkfunc(p,W1_0,b1_0,W2_0,b2_0,alpha)
# W1_1,b1_1,W2_1,b2_1,error,mse=stochnetworkfunc(p,W1_0,b1_0,W2_0,b2_0,alpha)
# #print(error,mse)
#
# # ## i. Plot the trained networks with the network outputs. Compare them.
# tt=np.zeros((len(p)))
# for i in range(0,len(p)):
#     z1=np.dot(W1_1,p[i])
#     a1=logsig(z1+b1_1)
#     a2=np.dot(W2_1,a1)+b2_1
#     tt[i]=(a2)
#
# plt.plot(p,t,label='Original function',color='green')
# plt.plot(p,tt,label='Network function',color='blue')
# plt.legend()
# plt.show()
#
# # ## ii.Plot squared error for each epochs.
# plt.loglog(np.arange(0,epochs),mse)
# plt.show()

# ## iv. Implement Batch approach (True Gradient) and repeat part i and ii.
alpha=.01
p=np.linspace(-2,2,200)
epochs=50000

def batchnetworkfunc(p,W1_0,b1_0,W2_0,b2_0,alpha,t):
    mse=np.zeros((epochs))
    for j in range(0,epochs):
        e1 = np.zeros(( len(p)))
        r1=np.zeros((1,S1))
        r2=np.zeros((S1,1))
        ss1=0
        ss2=0
        for i in range(len(p)):
            n=np.dot(W1_0, p[i]) + b1_0
            a1_1 =logsig(n)
            a2_1=((np.dot(W2_0, a1_1)) + b2_0)
            e1[i] = g(p[i]) - a2_1
            s2=(-2)*1*e1[i]
            s1=np.dot(np.diag(((1-a1_1)*a1_1).flatten()),(W2_0).T)*s2
            ss1=ss1+s1
            ss2 = ss2 + s2
            rr=np.dot(s2,a1_1.T)
            r1=np.add(r1,rr)
            r2+=s1*p[i]
        W2_0 = W2_0 - ((alpha/len(p))*r1)
        W1_0 = W1_0 - ((alpha/len(p))*r2)
        b2_0 = b2_0 - (alpha/len(p))*ss2
        b1_0= b1_0 -(alpha/len(p))*ss1
        mse[j]=np.dot(e1.T,e1)
    return np.array(W1_0), np.array(b1_0),np.array(W2_0), np.array(b2_0),e1,mse

#batchnetworkfunc(p,W1_0,b1_0,W2_0,b2_0,alpha,t)
W1_1b,b1_1b,W2_1b,b2_1b,errorb,mseb=batchnetworkfunc(p,W1_0,b1_0,W2_0,b2_0,alpha,t)
#print(errorb,mseb)

# ## i. Plot the trained networks with the network outputs. Compare them.
ttb=np.zeros((len(p)))
for i in range(0,len(p)):
    z1b=np.dot(W1_1b,p[i])
    a1b=logsig(z1b+b1_1b)
    a2b=np.dot(W2_1b,a1b)+b2_1b
    ttb[i]=(a2b)

plt.plot(p,t,label='Original function',color='green')
plt.plot(p,ttb,label='Network function',color='blue')
plt.legend()
plt.show()

# ## ii.Plot squared error for each epochs.
plt.loglog(np.arange(0,epochs),mseb)
plt.show()


