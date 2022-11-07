import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import preReq



df = pd.read_csv('penguins.csv')
df['gender'] = LabelEncoder().fit_transform(df['gender'])
scaled_df = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df.iloc[:,1:]), columns=df.iloc[:,1:].columns)
scaled_df['species'] = df['species']

selectedF = (preReq.INarr[0], preReq.INarr[1])
selectedC = (preReq.INarr[2], preReq.INarr[3])
weight = np.random.random((2))
eta = preReq.INarr[4]
epochs = preReq.INarr[5]
bias = np.random.randn() if preReq.INarr[6] else 0


# Visualizations
def visualizeData(scaled_df):  # 10 combs
    for i in range(len(preReq.Features)-1):
        for j in range(i+1, len(preReq.Features)):
            fL1, fL2 = preReq.Features[i], preReq.Features[j]
            f1, f2 = scaled_df[fL1], scaled_df[fL2]
            
            plt.xlabel(fL1)
            plt.ylabel(fL2)

            CF = [(f1[:50], f2[:50]), (f1[50:100], f2[50:100]), (f1[100:], f2[100:])]
            for plot in range(3):
                plt.scatter(CF[plot][0], CF[plot][1])

            plt.legend([preReq.Classes[0], preReq.Classes[1], preReq.Classes[2]])
            plt.show()
            
#visualizeData(scaled_df)


# Main

misClass = [x for x in preReq.Classes if x not in selectedC]
misFeatures = [x for x in preReq.Features if x not in selectedF]

scaled_df.drop(scaled_df.index[(scaled_df["species"] == misClass[0])], axis=0, inplace=True)
scaled_df.drop(columns=misFeatures, axis=1, inplace=True)
scaled_df['species'] = scaled_df['species'].replace(selectedC, [1, -1])


C1, C2 = (scaled_df[:50]).sample(frac=1), (scaled_df[50:100]).sample(frac=1)
C1_train, C1_test, C2_train, C2_test = C1[:30], C1[30:], C2[:30], C2[30:]

all_train_data = pd.concat([C1_train, C2_train]).to_numpy()
all_test_data = pd.concat([C1_test, C2_test]).to_numpy()

train_data, train_target = all_train_data[:,:2], all_train_data[:, 2:]
test_data, test_target = all_test_data[:,:2], all_test_data[:, 2:]




def PerceptronAlgo(weight, bias):
    for epoch in range(epochs):
        predict = (train_data.dot(weight.transpose()) + bias)
        yPredTrain = np.where(predict>0, 1, -1)
        for i in range(len(train_target)):
            if yPredTrain[i] != train_target[i]:
                loss = train_target[i] - yPredTrain[i]
                weight += eta*loss*train_data[i]
                bias += eta*loss if bias else 0
    return yPredTrain

def classifyTrain():
    plt.figure("Trained Features Figure")
    plt.scatter(x=train_data[:, :1], y=train_data[:, 1:2], c=train_target)
    plt.plot(np.array([0, 1]), np.array([(-bias)/weight[1], (-weight[0]-bias)/weight[1]]))
    plt.show()

yPredTrain = PerceptronAlgo(weight, bias)
classifyTrain()



# Testing
ytest=(test_data.dot(weight.transpose()) + bias)
yPredTest = np.where(ytest>0, 1, -1)

def ConfuionMatrix(target, yPred):
    row = list()
    confMat = np.zeros([2, 2])
    
    for i in range(len(yPred)):
        if target[i] == 1:
            if yPred[i] == 1:
                confMat[0][0] += 1
                row.append([target[i], yPred[i], 'T'])
            else:
                confMat[0][1] += 1
                row.append([target[i], yPred[i], 'F'])

        elif target[i] == -1:
            if yPred[i] == -1:
                confMat[1][1] += 1
                row.append([target[i], yPred[i], 'T'])
            else:
                confMat[1][0] += 1
                row.append([target[i], yPred[i], 'F'])

    return ((np.trace(confMat)/np.sum(confMat))*100), confMat

trainAcc, confMat = ConfuionMatrix(train_target, yPredTrain)
testAcc, confMat = ConfuionMatrix(test_target, yPredTest)
print("Training",trainAcc)
print("Testing",testAcc)

preReq.OUTarr[0] = selectedC[0] if sum(yPredTest) else selectedC[1]
preReq.OUTarr[1] = trainAcc
preReq.OUTarr[2] = testAcc
preReq.OUTarr[3][0] = confMat[0][0]
preReq.OUTarr[3][1] = confMat[0][1]
preReq.OUTarr[3][2] = confMat[1][0]
preReq.OUTarr[3][3] = confMat[1][1]
