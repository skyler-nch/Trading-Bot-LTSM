import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

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
testdatasize = 0.1*len(rawdata)
