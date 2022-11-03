#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('penguins.csv')
df
W = [random.random(),random.random()]

def signum(V):
    if V>=0:
        return 1
    else:
        return 0

def Model(eboch,x,b,eta):
    for i in range(eboch):
        y_i =signum(np.dot(W[i].T,x[i])+b)
        if y_i != t_i:
            Loss = (t_i-y_i)
            W[i+1] = W[i]+np.dot((np.dot(eta,Loss)),x[i])
        else:
            continue

