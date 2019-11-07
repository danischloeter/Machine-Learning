
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
from numpy import linalg as LA
import sympy


# ## 2-   Expand x  in terms of the following basis set. (Verify your answer using Python.)

# In[22]:


A=np.array([1,2,2])


# In[23]:


v1=np.array([-1,1,0])
v2=np.array([1,1,-2])
v3=np.array([1,1,0])


# In[24]:


Q=np.array([v1,v2,v3]).T


# In[25]:


Q


# In[26]:


v_b=np.linalg.solve(Q,A)
print('The vector in the new coordinates \n v_B={}'.format(v_b))


# ## 5- iv)

# ### ii) Find the eigenvalues and eigenvectors of the transformation.

# In[27]:


A=np.array([[1,1],[-1,1]])
w, v = LA.eig(A)
print('eigenvalues '+str(w))
print('eigenvectors '+str(v))


# For each result the eigen values are the same as the written calculations but the eigenvectors vary. This is explained by the fact that python normalizes while we don't

# ### iii) Find the matrix representation for A relative to the eigenvectors as the basis vectors

# In[28]:


B = v


# In[29]:


y = np.linalg.inv(B) 


# In[31]:


C= np.matmul(y,A)
Aprime= np.matmul(T, B)
Aprime

