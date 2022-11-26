#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def activation(val, activeFn):
    if activeFn == 'Sigmoid':
        return 1/(1 + np.exp(-val))
    else:
        return np.tanh(val)


# In[3]:


def BackPropagationAlgo(inN, outN, weights, activeFn, epochs, eta, layers, nn, bias):
    import numpy as np

    neurons = list()
    for n in nn:
        neurons.append(np.zeros(n))
    neurons.append(np.zeros(3))
    holder = neurons
    
    for epoch in range(epochs):
        for itr in range(len(inN)):
            for layer in range(layers+1):  #Feed Forward
                if layer == 0:
                    neurons[layer] = activation(inN[itr].dot(weights[layer]), activeFn)
                    continue
                neurons[layer] = activation(neurons[layer-1].dot(weights[layer]), activeFn)
            for n in range(len(neurons)-1):
                neurons[n][0] = bias
            
            for layer in range(layers, -1, -1):  #Back Propagate
                goBack = neurons[layer] * (1-neurons[layer])
                if layer == layers:
                    err = outN[itr] - neurons[-1]
                    holder[layer] = err * goBack
                    continue
                holder[layer] = goBack * (holder[layer+1].dot(weights[layer+1].transpose()))
            
            for layer in range(layers+1):  #Update Weights
                (x, y) = weights[layer].shape
                if layer == 0:
                    inNew = inN[itr].reshape(x, 1).dot(holder[layer].reshape(1, y))
                    weights[layer] += (eta * inNew)
                    continue
                inNew = neurons[layer-1].reshape(x, 1).dot(holder[layer].reshape(1, y))
                weights[layer] += (eta * inNew)
            
    return weights

