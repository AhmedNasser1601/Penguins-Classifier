import numpy as np
import pandas as pd
import warnings


def activation(val, activeFn):
    if activeFn == 'Sigmoid':
        warnings.filterwarnings('ignore')
        return 1/(1 + np.exp(-val))
    else:
        return np.tanh(val)


def BackPropagationAlgo(inTrain, inTest, outTrain, weights, activeFn, epochs, eta, layers, nn, bias):
    import numpy as np

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
                    neurons[0] = activation(inTrain[itr].dot(weights[0]), activeFn)
                    continue
                neurons[layer] = activation(neurons[layer-1].dot(weights[layer]), activeFn)
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
    
    # Testing & Predict Output
    fOut = list()
    for i in range(len(inTest)):
        for layer in range(layers+1):  #Feed Forward
            if layer == 0:
                final[layer] = activation(inTest[i].dot(weights[layer]), activeFn)
                continue
            final[layer] = activation(final[layer-1].dot(weights[layer]), activeFn)
        for n in range(len(final)-1):
            final[n][0] = bias
        fOut.append(final[-1])

    for y in fOut:
        maximum = np.argmax(y)
        for idx in range(len(y)):
            y[idx] = 1 if idx==maximum else 0
    
    return fOut


def ConfusionMatrix(targetTest, predOut):
    case = list()
    confMat = np.zeros([3, 3])
    classes = ['Adelie', 'Gentoo', 'Chinstrap']
    truthVals = pd.DataFrame(columns=['Pred', 'Real', 'Match'])

    for idx in range(len(targetTest)):
        confMat[np.argmax(targetTest[idx])][np.argmax(predOut[idx])] += 1
        case.append([
            classes[np.argmax(predOut[idx])],
            classes[np.argmax(targetTest[idx])],
            '[*]' if np.argmax(targetTest[idx])==np.argmax(predOut[idx]) else '[ ]']
        )
    
    truthVals = truthVals.append(pd.DataFrame(case, columns=['Pred', 'Real', 'Match']), ignore_index=True)
    
    print(
        '\n   |> Testing Evaluation <| \n',
        '    Overall Acc: %{0:.2f}'.format((np.trace(confMat)/np.sum(confMat))*100),
        '\n\n', truthVals,
        '\n\n', '-----> Confusion Matrix <-----\n',
        pd.DataFrame(confMat, columns=['C1', 'C2', 'C3'], index=['C1', 'C2', 'C3']), '\n'
    )

    for i in range(len(classes)):
        print('{}: %{}'.format(classes[i], ((confMat[i][i]/(len(targetTest)/len(classes)))*100)))
    print('\n', '-'*50)