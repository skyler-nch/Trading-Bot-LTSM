from alpha_vantage.timeseries import TimeSeries
import os
# API KEY-------------------------------------------------
key = "LOK9IWR45QLHXH88"
ts = TimeSeries(key)
#---------------------------------------------------------


# Handling Historical Input------------------------------------------

market = ["GOOGL", "DJI"]
#USA eastern time by default (GMT -5), 9:30AM opens, 4PM closes, Weekend closes
#if historical is true, it will fetch ALL data past 20 years for every stock in the market
#---------------------------
historical = True
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
#---------------------------------------------------------


