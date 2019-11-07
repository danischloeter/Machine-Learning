#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
from numpy import linalg as LA
import sympy
import matplotlib.pyplot as plt


# # E.6: Python Exercise
# The vectors in the ordered set defined below were obtained by measuring the weight and ear lengths of toy rabbits and bears in the Fuzzy Wuzzy Animal Factory. The target values indicate whether the respective input vector was taken from a rabbit (0) or a bear (1). The first element of the input vector is the weight of the toy, and the second element is the ear length.
# 
# ## i. i. Use Python to initialize and train a network to solve this ”practical” problem.

# In[65]:


# Defining my 8 inputs
p= (np.array([1,4]),np.array([1,5]),np.array([2,4]),np.array([2,5]), np.array([3,1]), np.array([3,2]), np.array([4,1]), np.array([4,2]))
# Defining my 8 targets
t = [0,0,0,0,1,1,1,1]


# In[66]:


# First I will create my hardlimfunction which returns 1 for values greater than or equal to 0 and 0 else.

def hardlimfunc(n):
    if n.sum() >= 0:
        a = 1
    else:
        a = 0
    return a


# In[67]:


p


# In[68]:


# Now I will define my network
w_old = np.matrix([[0.5, 1]])
b_old = np.array([[0]])

def networkfunc(p,w_old,b_old):
    for j in range(5):
        n = []
        for i in range(len(p)):
            print("Outer iteration:", j, " Inner iteration:", i)
            n.append(((np.dot(w_old, p[i])) + b_old))
            print("Input (p):", p[i])
            a = hardlimfunc(n[i])
            print("My predicted target: ",a)
            print("My real target: ",t[i])
            e = t[i] - a
            w_old = w_old + (np.dot(e, p[i].T))
            b_old = b_old + e
            print("Error:", e)
            print('w new is: ',w_old)
            print('b new is',b_old)
        #If the error is 0 after passing all my inputs once then end the for loop
        if e == 0:
            print('Break')
            break

    return np.array(w_old), np.array(b_old)


# In[69]:


w1,b = networkfunc(p,w_old,b_old)
w = np.array(w1).T


# ## ii. Use Python test the resulting weight and bias values against the input vectors.

# In[70]:


networkfunc(p,w1,b)


# As we can observe when we ran our network function again but using the new weight matrix and b vector we obtain 0 error for all inputs and the same weight matrix and b vector as result

# ## iii. Please plot the inputs and check your trained weight vector and validate your results by plotting the trained weight and bias.

# In[71]:


# p_1 and p_2 create my decision boundary, as the decision is made on 0, then WP+b=0
p_1 = np.linspace(0,4)
p_2 = ((-w[0] * p_1) - b)/ w[1]
plt.figure()
for i in range(len(p)):
    # Plotting each input
    plt.scatter(p[i][0],p[i][1])
    # This shows my weight vector in the plot
    # 2,2.5 indicates the starting point for my weight vector and it is in the center of my decision Boundary
    plt.quiver(2,2.5,w[0],w[1])
plt.plot(p_1, p_2.T)
plt.show()


# In[ ]:





# In[ ]:




