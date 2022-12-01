import numpy as np
import pandas as pd
import warnings


def activationFn(val, activeFn):
    if activeFn == 'Sigmoid':
        warnings.filterwarnings('ignore')
        return 1/(1 + np.exp(-val))
    else:
        warnings.filterwarnings('ignore')
        return np.tanh(val)


def BackPropagationFn(inTrain, inTest, outTrain, weights, activeFn, epochs, eta, layers, nn, bias):
    neurons = list()
    for n in nn:
        neurons.append(np.zeros(n))
    neurons.append(np.zeros(3))
    holder = neurons
    final = neurons
    
    for epoch in range(epochs):
        for itr in range(len(inTrain)):
            for layer in range(layers+1):  #Feed Forward
                if layer == 0:
                    neurons[0] = activationFn(inTrain[itr].dot(weights[0]), activeFn)
                    continue
                neurons[layer] = activationFn(neurons[layer-1].dot(weights[layer]), activeFn)
            for n in range(len(neurons)-1):
                neurons[n][0] = bias
            
            for layer in range(layers, -1, -1):  #Back Propagate
                goBack = neurons[layer] * (1-neurons[layer])
                if layer == layers:
                    err = outTrain[itr] - neurons[-1]
                    holder[layer] = err * goBack
                    continue
                holder[layer] = goBack * (holder[layer+1].dot(weights[layer+1].transpose()))
            
            for layer in range(layers+1):  #Update Weights
                (x, y) = weights[layer].shape
                if layer == 0:
                    inNew = inTrain[itr].reshape(x, 1).dot(holder[0].reshape(1, y))
                    weights[layer] += (eta * inNew)
                    continue
                inNew = neurons[layer-1].reshape(x, 1).dot(holder[layer].reshape(1, y))
                weights[layer] += (eta * inNew)
    
    outPred = list()
    for input in (inTrain, inTest):
        fOut = list()
        for i in range(len(input)):
            for layer in range(layers+1):  #Feed Forward
                if layer == 0:
                    final[layer] = activationFn(input[i].dot(weights[layer]), activeFn)
                    continue
                final[layer] = activationFn(final[layer-1].dot(weights[layer]), activeFn)
            for n in range(len(final)-1):
                final[n][0] = bias
            fOut.append(final[-1])

        for y in fOut:
            maximum = np.argmax(y)
            for idx in range(len(y)):
                y[idx] = 1 if idx==maximum else 0
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
    
    print(
        '    Overall Acc: %{0:.2f}'.format((np.trace(confMat)/np.sum(confMat))*100),
        '\n\n', truthVals,
        '\n\n', '-----> Confusion Matrix <-----\n',
        pd.DataFrame(confMat, columns=['C1', 'C2', 'C3'], index=['C1', 'C2', 'C3']), '\n'
    )

    for i in range(len(classes)):
        print('{}: %{}'.format(classes[i], ((confMat[i][i]/(len(target)/len(classes)))*100)))