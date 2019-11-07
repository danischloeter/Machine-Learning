#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from sympy import *

x1 = symbols('x1')
x2 = symbols('x2')
fx1=36*(x1**3)+18*x2*(x1**2)-270*(x1**2)-22*x1*(x2**2)+60*x1*x2+470*x1-4*(x2**3)+80*(x2**2)-310*x2-10
fx2=6*(x1**3)-22*(x1**2)*x2+30*(x1**2)-12*x1*(x2**2)+160*x1*x2-310*x1+16*(x2**3)-120*(x2**2)+210*x2-10
e1 = solve([fx1,fx2],[x1,x2])

print(e1)


# In[ ]:




