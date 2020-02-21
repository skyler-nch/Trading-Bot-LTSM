#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
import os
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

#we will train for now, the high of the stock price, also to inverse the records from oldest to newest

traindata = stockdata["high"][::-1][:-testdatasize]
testdata = stockdata["high"][::-1][-testdatasize:]
print("training data consists of {} records, testing data consists of {} records".format(len(traindata),len(testdata)))


#normalise the training data
scaler = MinMaxScaler(feature_range=(-1,1))
#print(scaler)

#convert to numpy array, then reshape to normalise
NormalisedTrainingData = scaler.fit_transform(np.asarray(traindata).reshape(-1,1))

#convert the dataset to FloatTensor object fot the pytorch model
NormalisedTrainingData = torch.FloatTensor(NormalisedTrainingData).view(-1)

#any sequence window will do, but since its daily data, its nicer to have it at same size of the testing data
trainingwindow = testdatasize
trainingsequence = CreateInOutSequence(NormalisedTrainingData,trainingwindow)

#print(trainingsequence[:5])

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#Training Model
#-----------------------------------------------
epochs = 150
#-----------------------------------------------
#print(trainingsequence)
for i in range(epochs):
    for seq, labels in trainingsequence:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


#Prediction stage
PredictionCount = testdatasize

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
print(previousvalue)

correct = 0
for i in range(len(actual_predictions)):
    if float(previousvalue[i]) == float(testdata[i]):
        testcompare = "Same as previous"
    elif float(previousvalue[i]) < float(testdata[i]):
        testcompare = "Higher then previous"
    else:
        testcompare = "Lower then previous"

    if float(previousvalue[i]) == actual_predictions[i]:
        predictcompare = "Same as previous"
    elif float(previousvalue[i]) < actual_predictions[i]:
        predictcompare = "Higher then previous"
    else:
        predictcompare = "Lower then previous"

    if testcompare == predictcompare:
        correct += 1
        
    print("Actual Results: {},{}    Predicted Result: {},{}    Difference: {}".format(testdata[i], testcompare, actual_predictions[i], predictcompare,actual_predictions[i]-float(testdata[i])))

print("Prediction on High/Low is {}/{} correct".format(correct, len(actual_predictions)))
