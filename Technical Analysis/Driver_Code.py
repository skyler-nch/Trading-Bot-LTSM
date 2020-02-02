from alpha_vantage.timeseries import TimeSeries
# API KEY------------------
key = 'LOK9IWR45QLHXH88'
ts = TimeSeries(key)
#--------------------------

data,meta_data = ts.get_intraday('GOOGL')
print(data)
for item in data:
    print(item)
