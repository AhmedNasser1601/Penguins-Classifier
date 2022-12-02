import numpy as np
import pandas as pd
import warnings


def activationFn(val, activeFn):
    warnings.filterwarnings('ignore')
    if activeFn == 'Sigmoid': return 1/(1+np.exp(-val))
    else: return np.tanh(val)


def BackPropagationFn(inTrain, inTest, outTrain, weights, activeFn, epochs, eta, layers, nn, bias):
    neurons, outPred = list(), list()
    for n in nn:
        neurons.append(np.zeros(n))
    neurons.append(np.zeros(3))
    holder, final = neurons, neurons
    
    for epoch in range(epochs):
        for itr in range(len(inTrain)):
            neurons[0] = activationFn(inTrain[itr].dot(weights[0]), activeFn)
            for layer in range(1, layers+1):  #Feed Forward
                neurons[layer] = activationFn(neurons[layer-1].dot(weights[layer]), activeFn)
            for n in range(len(neurons)-1):
                neurons[n][0] = bias
            
            holder[layers] = (outTrain[itr]-neurons[-1]) * (neurons[layers]*(1-neurons[layers]))
            for layer in range(layers-1, -1, -1):  #Back Propagate
                holder[layer] = (neurons[layer]*(1-neurons[layer])) * (holder[layer+1].dot(weights[layer+1].transpose()))
            
            (x, y) = weights[0].shape
            weights[0] += (eta * (inTrain[itr].reshape(x, 1).dot(holder[0].reshape(1, y))))
            for layer in range(1, layers+1):  #Update Weights
                (x, y) = weights[layer].shape
                weights[layer] += (eta * (neurons[layer-1].reshape(x, 1).dot(holder[layer].reshape(1, y))))
    
    for input in (inTrain, inTest):
        fOut = list()
        for itr in range(len(input)):
            final[0] = activationFn(input[itr].dot(weights[0]), activeFn)
            for layer in range(1, layers+1):  #Feed Forward
                final[layer] = activationFn(final[layer-1].dot(weights[layer]), activeFn)
            for n in range(len(final)-1):
                final[n][0] = bias
            fOut.append(final[-1])
        for y in fOut:
            for idx in range(len(y)):
                y[idx] = 1 if idx==np.argmax(y) else 0
        outPred.append(fOut)
    return outPred


def ConfusionMatrixFn(target, predOut):
    case = list()
    confMat = np.zeros([3, 3])
    classes = ['Adelie', 'Gentoo', 'Chinstrap']
    truthVals = pd.DataFrame(columns=['Pred', 'Real', 'Match'])

    for idx in range(len(target)):
        confMat[np.argmax(target[idx])][np.argmax(predOut[idx])] += 1
        case.append([
            classes[np.argmax(predOut[idx])],
            classes[np.argmax(target[idx])],
            '[*]' if np.argmax(target[idx])==np.argmax(predOut[idx]) else '[ ]']
        )
    
    truthVals = truthVals.append(pd.DataFrame(case, columns=['Pred', 'Real', 'Match']), ignore_index=True)
    
    print('    Overall Acc: %{0:.2f}'.format((np.trace(confMat)/np.sum(confMat))*100),
          '\n\n', truthVals, '\n\n', '-----> Confusion Matrix <-----\n',
          pd.DataFrame(confMat, columns=['C1', 'C2', 'C3'], index=['C1', 'C2', 'C3']), '\n')

    for i in range(len(classes)):
        print('{}: %{}'.format(classes[i], ((confMat[i][i]/(len(target)/len(classes)))*100)))
