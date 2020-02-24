#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
import os
import time
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def CreateInOutSequence(inputlist,trainingwindow):
    sequence = []
    for i in range(len(inputlist)-trainingwindow):
        train_seq = inputlist[i:i+trainingwindow]
        train_label = inputlist[i+trainingwindow:i+trainingwindow+1]
        sequence.append((train_seq,train_label))
    return sequence


#Input######################################
inputcsv = "DJI.txt"

#how many lines of data are used
datasize = 2000

#percentage dictates how much of the latest records will be used as test data
percentage = 0.1

# open, high, low, close, adjusted close, volume are available for use
datatype = "high"

#lr is learning rate
lr = 0.001

#epoch determines how many rounds does the network goes through
epochs = 150

#the sliding window cutout for the neural network
trainingwindow = 7

#how many sequences should be predicted, best to follow values by the training window
PredictionCount = trainingwindow

############################################


listofitems = os.listdir("{}\\Input_CSV\\Historical".format(os.path.dirname(__file__)))
print("There are currently {} in the input folder".format(str(listofitems)))


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

#we will train for now, the high of the stock price, also to inverse the records from oldest to newest


testdatasize = round(percentage*datasize)
data = stockdata[datatype][::-1][:datasize]
traindata = data[:-testdatasize]
testdata = data[-testdatasize:]
print("training data consists of {} records, testing data consists of {} records".format(len(traindata),len(testdata)))


#normalise the training data
scaler = MinMaxScaler(feature_range=(-1,1))
#print(scaler)

#convert to numpy array, then reshape to normalise
NormalisedTrainingData = scaler.fit_transform(np.asarray(traindata).reshape(-1,1))

#convert the dataset to FloatTensor object fot the pytorch model
NormalisedTrainingData = torch.FloatTensor(NormalisedTrainingData).view(-1)

#any sequence window will do, its nicer to have it at same size of the testing data

trainingsequence = CreateInOutSequence(NormalisedTrainingData,trainingwindow)

#print(trainingsequence[:5])

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)


#Training Model
#-----------------------------------------------
#-----------------------------------------------
#print(trainingsequence)
previousloss = 1
for i in range(epochs):
    start = time.time()
    for seq, labels in trainingsequence:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if single_loss.item() < previousloss:
        leastlossmodel = model.state_dict()
        leastlossvalue = single_loss.item()
        previousloss = leastlossvalue

    stop = time.time()
    print("epoch: {}/{} \t loss: {} \t time taken: {} seconds".format(i+1,epochs,single_loss.item(),round(stop-start,4)))

torch.save(model.state_dict(), "{}\\Models\\{}".format(os.path.dirname(__file__),"{}-{}.pth".format(inputcsv.split(".")[0],leastlossvalue)))
print("saved model with {} loss value".format(leastlossvalue))



#Prediction stage

test_inputs = NormalisedTrainingData[-PredictionCount:].tolist()
#print(test_inputs)

model.eval()

for i in range(PredictionCount):
    seq = torch.FloatTensor(test_inputs[-trainingwindow:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())


predicted_values = test_inputs[PredictionCount:]
actual_predictions = scaler.inverse_transform(np.array(test_inputs[PredictionCount:] ).reshape(-1, 1))
actual_predictions = [item[0] for item in actual_predictions]
previousvalue = [traindata[-1]]+testdata
#print(previousvalue)
print("##########################Results##########################")
correct = 0
for i in range(len(actual_predictions)):
    if float(previousvalue[i]) == float(testdata[i]):
        testcompare = "Same"
    elif float(previousvalue[i]) < float(testdata[i]):
        testcompare = "Higher"
    else:
        testcompare = "Lower"

    if float(previousvalue[i]) == actual_predictions[i]:
        predictcompare = "Same"
    elif float(previousvalue[i]) < actual_predictions[i]:
        predictcompare = "Higher"
    else:
        predictcompare = "Lower"

    if testcompare == predictcompare:
        correct += 1
        
    print("Actual Results: {},{}\t\tPredicted Result: {},{}\t\tDifference: {}".format(round(float(testdata[i]),2), testcompare, round(actual_predictions[i],2), predictcompare,round(actual_predictions[i]-float(testdata[i]),2)))

print("Prediction on High/Low is {}/{} correct".format(correct, len(actual_predictions)))
