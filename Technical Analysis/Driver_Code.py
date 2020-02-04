from alpha_vantage.timeseries import TimeSeries
import os
# API KEY------------------
key = "LOK9IWR45QLHXH88"
ts = TimeSeries(key)
#--------------------------


# Handling Input-----------

market = ["GOOGL"]
#USA eastern time by default (GMT -5), 9:30AM opens, 4PM closes, Weekend closes

for stocks in market:
    data,meta_data = ts.get_intraday(stocks)
    f = open("{}\\Input_CSV\\{}.txt".format(os.path.dirname(__file__),str(stocks)),'w')

    for days in data:
        print(data[days])
        currentday = data[days]
        
        # Use this if viewer friendly text is needed
        #currentline = "0. day: {}\t1. open: {}\t2. high: {}\t3. low: {}\t4. close: {}\t5. volume: {}\n"

        #Use this if CSV version is needed
        currentline = "{},{},{},{},{},{}\n"

        
        currentline = currentline.format(days,currentday["1. open"],currentday["2. high"],currentday["3. low"],currentday["4. close"],currentday["5. volume"],)

        f.write(currentline)
        
        
f.close()
#--------------------------


