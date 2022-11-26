#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter.font as tkFont

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from Model import *


# In[2]:


def genLblTxt(text):
    lbl, txt = StringVar(), Entry(Top)
    lbl.set(text)
    Label(Top, textvariable=lbl).pack(padx=5, pady=5)
    txt.pack(padx=5, pady=5)
    return txt

def getter():
    return (
        int(txtEnt[0].get()),
        [int(nn) for nn in (txtEnt[1].get()).split()],
        float(txtEnt[2].get()),
        int(txtEnt[3].get()),
        int(txtEnt[4].get()),
        txtEnt[5].get()
    )


# ---

# In[3]:


def modelRUN():
    df = pd.read_csv('penguins.csv')
    df['gender'] = LabelEncoder().fit_transform(df['gender'])    
    df = pd.DataFrame(ColumnTransformer([(df.columns[0], OneHotEncoder(),
                        [df.columns.get_loc(df.columns[0])])],
                      remainder='passthrough').fit_transform(df),
                      columns=['C1', 'C2', 'C3', 'X1', 'X2', 'X3', 'X4', 'X5'])
    
    try: layers, nn, eta, epochs, bias, activeFn = getter()
    except: layers, nn, eta, epochs, bias, activeFn = 2, (8, 4), 0.01, 100, 1, 'Sigmoid'
    
    df.insert(loc=3, column='bias', value=[bias for _ in range(len(df))])
        
    C1, C2, C3 = df[:50].sample(frac=1), df[50:100].sample(frac=1), df[100:].sample(frac=1)
    trainData = (pd.concat([C1[:30], C2[:30], C3[:30]])).to_numpy()
    testData = (pd.concat([C1[30:], C2[30:], C3[30:]])).to_numpy()
    
    inMat, outMat = np.zeros([len(trainData), 5+1]), np.zeros([len(trainData), 3])
    for i in range(len(trainData)):
        outMat[i], inMat[i] = trainData[i][0:3], trainData[i][3:]
    
    weights = list()
    weights.append(np.random.randn(5+1, nn[0]))
    for i in range(layers-1):
        weights.append(np.random.randn(nn[i], nn[i+1]))
    weights.append(np.random.randn(nn[-1], 3))
    
    weights = BackPropagationAlgo(inMat, outMat, weights, activeFn,
                                            epochs, eta, layers, nn, bias)
    
    print(weights)
    
modelRUN()


# In[4]:


def PredictFn():
    for i in range(len(txtEnt)):
        if txtEnt[i].get() in ('', 'Activation Function >>'):
            messagebox.showerror(title="error", message="Insert the missing inputs", parent=Top)
            break
    else:
        print(getter())
        modelRUN()


# ---

# In[5]:


Top = Tk()
Top.geometry('300x400')
Top.title('Back Propagation Algo')
Top.resizable(False, False)

txtEnt = [0]*6
txtEnt[0] = (genLblTxt('No. of Hidden Layers'))
txtEnt[1] = (genLblTxt('No. of Neurons\n(separate by space)'))
txtEnt[2] = (genLblTxt('Learning Rate Value'))
txtEnt[3] = (genLblTxt('No. of Epochs'))

txtEnt[4] = IntVar()
cb_obj = Checkbutton(Top, text='Add Bias', variable=txtEnt[4], onvalue=1, offvalue=0)
cb_obj.pack()

txtEnt[5] = ttk.Combobox(Top, textvariable=StringVar(), state='readonly', values=('Sigmoid', 'TanH'))
txtEnt[5].set('Activation Function >>')
txtEnt[5].pack(padx=5, pady=5)

btn_submit = Button(Top, text='Run', width=10, command=PredictFn, bg='green', fg='yellow')
btn_submit.pack(pady=25)

Top.mainloop()

