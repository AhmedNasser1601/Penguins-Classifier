import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import preReq




df = pd.read_csv('penguins.csv')




df['gender'] = LabelEncoder().fit_transform(df['gender'])
scaled_df = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df.iloc[:, 1:]), columns=df.iloc[:, 1:].columns)
scaled_df['species'] = df['species']



selectedF = (preReq.INarr[0], preReq.INarr[1])
selectedC = (preReq.INarr[2], preReq.INarr[3])
weight = np.random.random((2))
eta = preReq.INarr[4]
epochs = preReq.INarr[5]
bias = np.random.randn() if preReq.INarr[6] else 0
MeanS= preReq.INarr[7]


# # Visualizations



def visualizeData(scaled_df):  # 10 combs
    for i in range(len(preReq.Features) - 1):
        for j in range(i + 1, len(preReq.Features)):
            fL1, fL2 = preReq.Features[i], preReq.Features[j]
            f1, f2 = scaled_df[fL1], scaled_df[fL2]

            plt.xlabel(fL1)
            plt.ylabel(fL2)

            CF = [(f1[:50], f2[:50]), (f1[50:100], f2[50:100]), (f1[100:], f2[100:])]
            for plot in range(3):
                plt.scatter(CF[plot][0], CF[plot][1])

            plt.legend([preReq.Classes[0], preReq.Classes[1], preReq.Classes[2]])
            plt.show()


# visualizeData(scaled_df)


# # Main



misClass = list()
for x in preReq.Classes:
    if x not in selectedC:
        misClass.append(x)

misFeatures = list()
for x in preReq.Features:
    if x not in selectedF:
        misFeatures.append(x)



scaled_df.drop(scaled_df.index[(scaled_df["species"] == misClass[0])], axis=0, inplace=True)
scaled_df.drop(columns=misFeatures, axis=1, inplace=True)
scaled_df['species'] = scaled_df['species'].replace(selectedC, [1, -1])



C1, C2 = (scaled_df[:50]).sample(frac=1), (scaled_df[50:100]).sample(frac=1)
C1_train, C1_test, C2_train, C2_test = C1[:30], C1[30:], C2[:30], C2[30:]

all_train_data = pd.concat([C1_train, C2_train]).to_numpy()
all_test_data = pd.concat([C1_test, C2_test]).to_numpy()

train_data, train_target = all_train_data[:, :2], all_train_data[:, 2:]
test_data, test_target = all_test_data[:, :2], all_test_data[:, 2:]



def AdalineAlgo(epochs, weight, bias, eta, train_data, train_target,MeanS):
    import numpy as np

    for epoch in range(epochs):
        yPredTrain = (train_data.dot(weight.transpose()) + bias)
        for i in range(len(yPredTrain)):
            if yPredTrain[i] != train_target[i]:
                loss = train_target[i] - yPredTrain[i]
                weight += (eta * loss * train_data[i])
                if bias!=0:
                    bias = bias+(eta * loss)
        square_error = list()
        prediction = train_data.dot(weight.transpose()) + bias
        for i in range(len(train_target)):
            error = np.square(train_target[i] - prediction[i])
            square_error.append(error)
        MSE =  (1/len(square_error)) * (sum(square_error)/2)
        if MSE < float(MeanS):
            return yPredTrain
    return yPredTrain

yPredTrain = AdalineAlgo(epochs, weight, bias, eta, train_data, train_target,MeanS)




def classifyTrain():
    import matblotlip as plt
    plt.figure("Trained Features Figure")
    plt.scatter(x=train_data[:, :1], y=train_data[:, 1:2], c=train_target)
    plt.plot(np.array([0, 1]), np.array([(-bias) / weight[1], (-weight[0] - bias) / weight[1]]))
    plt.show()


#classifyTrain()


# # Testing



trainpred = train_data.dot(weight.transpose()) + bias
yPredTrain = np.where(trainpred>0, 1, -1)

testpred = test_data.dot(weight.transpose()) + bias
yPredTest = np.where(testpred>0, 1, -1)




def ConfuionMatrix(target, yPred):
    import numpy as np
    import pandas as pd

    testCase = list()
    confMat = np.zeros([2, 2])

    for i in range(len(yPred)):
        if target[i] == 1:
            if yPred[i] == 1:
                confMat[0][0] += 1
                testCase.append([int(target[i][0]), yPred[i], '[*]'])
            else:
                confMat[0][1] += 1
                testCase.append([int(target[i][0]), yPred[i], '[ ]'])

        elif target[i] == -1:
            if yPred[i] == -1:
                confMat[1][1] += 1
                testCase.append([int(target[i][0]), yPred[i], '[*]'])
            else:
                confMat[1][0] += 1
                testCase.append([int(target[i][0]), yPred[i], '[ ]'])

    Cases = pd.DataFrame(testCase, columns=['Real', 'Pred', 'Match'])
    return ((np.trace(confMat) / np.sum(confMat)) * 100), confMat, pd.DataFrame(Cases,
                                                                                columns=['Real', 'Pred', 'Match'])



preReq.OUTarr[1], confMatTrain, truthValsTrain = ConfuionMatrix(train_target, yPredTrain)
print('|> Truth Values for Training <|\n   Training Accuracy: ', preReq.OUTarr[1], '%\n', truthValsTrain, '\n',
      '-' * 35)

preReq.OUTarr[2], confMatTest, truthValsTest = ConfuionMatrix(test_target, yPredTest)
print('|> Truth Values for Testing <|\n   Testing Accuracy: ', preReq.OUTarr[2], '%\n', truthValsTest, '\n', '-' * 35)




preReq.OUTarr[0] = selectedC[0] if sum(yPredTest) else selectedC[1]
preReq.OUTarr[3][0] = confMatTest[0][0]
preReq.OUTarr[3][1] = confMatTest[0][1]
preReq.OUTarr[3][2] = confMatTest[1][0]
preReq.OUTarr[3][3] = confMatTest[1][1]

