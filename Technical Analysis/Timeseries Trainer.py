import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def CreateInOutSequence(inputlist,trainingwindow):
    sequence = []
    for i in range(len(inputlist)-trainingwindow):
        train_seq = inputlist[i:i+trainingwindow]
        train_label = inputlist[i+trainingwindow:i+trainingwindow+1]
        sequence.append((train_seq,train_label))
    return sequence


#Input------------------------------------
listofitems = os.listdir("{}\\Input_CSV\\Historical".format(os.path.dirname(__file__)))
print("There are currently {} in the input folder".format(str(listofitems)))

inputcsv = "DJI.txt"

print("we will be traning on the {}".format(inputcsv))

stockdata = {"date": [],"open": [],"high": [],"low": [],"close": [],"adjustedclose": [],"volume": [],"dividendamount": [],"splitcoefficient": []}
with open("{}\\Input_CSV\\Historical\\{}".format(os.path.dirname(__file__),inputcsv),'r') as file:
    rawdata = file.readlines()

for line in rawdata:
    line = line.strip().split(",")
    stockdata["date"].append(line[0])
    stockdata["open"].append(line[1])
    stockdata["high"].append(line[2])
    stockdata["low"].append(line[3])
    stockdata["close"].append(line[4])
    stockdata["adjustedclose"].append(line[5])
    stockdata["volume"].append(line[6])
    stockdata["dividendamount"].append(line[7])
    stockdata["splitcoefficient"].append(line[8])
print("{} records found in file".format(len(rawdata)))
    
#Preprocessing------------------------------------

#percentage dictates how much of the latest records will be used as test data
percentage = 0.1
testdatasize = round(percentage*len(rawdata))

#we will train for now, the high of the stock price
traindata = stockdata["high"][:-testdatasize]
testdata = stockdata["high"][-testdatasize:]
print("training data consists of {} records, testing data consists of {} records".format(len(traindata),len(testdata)))

#normalise the training data
scaler = MinMaxScaler(feature_range=(-1,1))
print(scaler)

#convert to numpy array, then reshape to normalise
NormalisedTrainingData = scaler.fit_transform(np.asarray(traindata).reshape(-1,1))

#convert the dataset to FloatTensor object fot the pytorch model
NormalisedTrainingData = torch.FloatTensor(NormalisedTrainingData).view(-1)

#any sequence window will do, but since its daily data, its nicer to have it at 365
trainingwindow = 365
trainingsequence = CreateInOutSequence(NormalisedTrainingData,trainingwindow)

print(trainingsequence[:5])











        
