#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import preReq


# In[2]:


df = pd.read_csv('penguins.csv')
df


# In[3]:


print(preReq.INarr)


# In[4]:


preReq.OUTarr[0] = 1
preReq.OUTarr[1] = 2
preReq.OUTarr[2] = 3
preReq.OUTarr[3][0] = 41
preReq.OUTarr[3][1] = 42
preReq.OUTarr[3][2] = 43
preReq.OUTarr[3][3] = 44
preReq.OUTarr[4] = 5


# In[5]:


print("\t\tModel Executed Successfully")

