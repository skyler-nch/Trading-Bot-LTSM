from alpha_vantage.timeseries import TimeSeries
import os
# API KEY---------------------------------------------------------
key = "LOK9IWR45QLHXH88"
ts = TimeSeries(key)
#---------------------------------------------------------


# Handling Historical Input------------------------------------------

market = ["DJI"]
marketstart = '09:31:00'
marketstop = '16:00:00'
#USA eastern time by default (GMT -5), 9:30AM opens, 4PM closes, Weekend closes
#if historical is true, it will fetch ALL data past 20 years for every stock in the market
#---------------------------
historical = False
#---------------------------
if historical == True:
    print("fetching historical data")
    print("will fetch the following stocks: {}".format(str(market)))
    for stocks in market:
        data,meta_data = ts.get_daily_adjusted(stocks,outputsize='full')
        f = open("{}\\Input_CSV\\Historical\\{}.txt".format(os.path.dirname(__file__),str(stocks)),'w')

        for days in data:
            currentday = data[days]
            
            # Use this if viewer friendly text is needed
            #currentline = "0. day: {}\t1. open: {}\t2. high: {}\t3. low: {}\t4. close: {}\t5. adjusted close: {}\t6. volume: {}\t7. dividend amount: {}\t8. split coefficient: {}\n"

            #Use this if CSV version is needed
            currentline = "{},{},{},{},{},{}\n"

            
            currentline = currentline.format(days,currentday["1. open"],currentday["2. high"],currentday["3. low"],currentday["4. close"],currentday["5. adjusted close"],currentday["6. volume"],currentday["7. dividend amount"],currentday["8. split coefficient"])

            f.write(currentline)
        print("{}'s retrieve operation is done".format(stocks))
            
            
    f.close()
else:
    print("not fetching historical data")
#---------------------------------------------------------

#Handling Recent Input---------------------------------------------------------
#if recent is true, it will fetch all data by the minute in the past 24 hour for every stock in the market
#---------------------------
recent = True
#---------------------------
if recent == True:
    print("fetching recent data")
    print("will fetch the following stocks: {}".format(str(market)))
    for stocks in market:
        data,meta_data = ts.get_intraday(stocks,interval = '1min',outputsize = 'full')
        print(len(data))

        savedate = []
        incompletedate = []
        for minute in data:
            currentminute = minute.split(" ")
            if currentminute[1] == marketstart:
                savedate.append(currentminute[0])
            elif currentminute[1] == marketstop:
                incompletedate.append(currentminute[0])

        if len(set(savedate)^set(incompletedate))>0:
            print("{} is incomplete, please run again after the market closes".format(str(set(savedate)^set(incompletedate))))
                

#INCOMPLETE----------------------
                    
            
            

            




