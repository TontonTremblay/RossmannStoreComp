

#This is the script used to extend the data set features


#to work with the data
import pandas as pd 
#to make pretty plot
import seaborn as sns
from datetime import datetime
import numpy as np





df = pd.read_csv("Data/train.csv")
#Some extra values
dfLoc = pd.read_csv("Data/store_states.csv")
dfStore = pd.read_csv('Data/store.csv')
dftest = pd.read_csv("Data/test.csv")

#find out missing information 
# print np.unique(dftest.isnull().values)
f = lambda x : dfLoc[dfLoc['Store']==x].values[0][1]

# print f(10)

# quit()

# Have to find a faster way to add the store location. 
# df['StoreLoc'] = df['Store'].map(f)


#Convert the a to 1
def f(x):
	if x is 0:
		return 0
	else:
		return 1 

df['StateHoliday'] = df['StateHoliday'].map(f)
dftest['StateHoliday'] = dftest['StateHoliday'].map(f)


#Convert the date to int of days. Year 2013-01-01 is the int 0. 
dayStart = datetime.strptime('2013-01-01',"%Y-%m-%d")
f = lambda x : (datetime.strptime(x,'%Y-%m-%d') - dayStart).days
df["NumbDays"] = df["Date"].map(f)
dftest["NumbDays"] = dftest["Date"].map(f)

#add the month in the feature: 
f = lambda x : int(x[5:7])
df['Month'] = df['Date'].map(f)
dftest['Month'] = dftest['Date'].map(f)

#Add log sales values
f = lambda x : np.log(x+1)
df['Saleslog'] = df['Sales'].map(f)





train = df.loc[df['Sales']>0]
#Add some simple features e.g. 
#MeanSalesDayWeek
data = train.groupby(['Store','DayOfWeek'])['Sales'].mean()
data = data.reset_index()
data.rename(columns={'Sales': 'MeanSalesDayOfWeek'},inplace=True)
df = pd.merge(df,data, on = ['Store','DayOfWeek'],left_index=True, right_index=True,how ='outer')
df.MeanSalesDayOfWeek.fillna(data.MeanSalesDayOfWeek.mean(),inplace=True)
df['MeanSalesDayOfWeeklog'] = df['MeanSalesDayOfWeek'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','DayOfWeek'],how="left")
dftest.MeanSalesDayOfWeek.fillna(data.MeanSalesDayOfWeek.mean(),inplace=True)
dftest['MeanSalesDayOfWeeklog'] = dftest['MeanSalesDayOfWeek'].map(f)

data = None


#MinSalesDayWeek
data = train.groupby(['Store','DayOfWeek'])['Sales'].min()
data = data.reset_index()
data.rename(columns={'Sales': 'MinSalesDayWeek'},inplace=True)
df = pd.merge(df,data, on = ['Store','DayOfWeek'], how ='left')
df.MinSalesDayWeek.fillna(data.MinSalesDayWeek.mean(),inplace=True)
df['MinSalesDayWeeklog'] = df['MinSalesDayWeek'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','DayOfWeek'],how="left")
dftest.MinSalesDayWeek.fillna(data.MinSalesDayWeek.mean(),inplace=True)
dftest['MinSalesDayWeeklog'] = dftest['MinSalesDayWeek'].map(f)


data = None

#MaxSalesDayWeek
data = train.groupby(['Store','DayOfWeek'])['Sales'].max()
data = data.reset_index()
data.rename(columns={'Sales': 'MaxSalesDayWeek'},inplace=True)
df = pd.merge(df,data, on = ['Store','DayOfWeek'],how ='left')
df.MaxSalesDayWeek.fillna(data.MaxSalesDayWeek.mean(),inplace=True)
df['MaxSalesDayWeeklog'] = df['MaxSalesDayWeek'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','DayOfWeek'],how="left")
dftest.MaxSalesDayWeek.fillna(data.MaxSalesDayWeek.mean(),inplace=True)
dftest['MaxSalesDayWeeklog'] = dftest['MaxSalesDayWeek'].map(f)


data = None

#MeanSalesPromoDayWeek
data = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].mean()
data = data.reset_index()
data.rename(columns={'Sales': 'MeanSalesPromoDayWeek'},inplace=True)
df = pd.merge(df,data, on = ['Store','DayOfWeek','Promo'],how ='left')
df.MeanSalesPromoDayWeek.fillna(data.MeanSalesPromoDayWeek.mean(),inplace=True)
df['MeanSalesPromoDayWeeklog'] = df['MeanSalesPromoDayWeek'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','DayOfWeek','Promo'],how="left")
dftest.MeanSalesPromoDayWeek.fillna(data.MeanSalesPromoDayWeek.mean(),inplace=True)
dftest['MeanSalesPromoDayWeeklog'] = dftest['MeanSalesPromoDayWeek'].map(f)


data = None

#MinSalesPromoDayWeek
data = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].min()
data = data.reset_index()
data.rename(columns={'Sales': 'MinSalesPromoDayWeek'},inplace=True)
df = pd.merge(df,data, on = ['Store','DayOfWeek','Promo'], how ='left')
df.MinSalesPromoDayWeek.fillna(data.MinSalesPromoDayWeek.mean(),inplace=True)
df['MinSalesPromoDayWeeklog'] = df['MinSalesPromoDayWeek'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','DayOfWeek','Promo'],how="left")
dftest.MinSalesPromoDayWeek.fillna(data.MinSalesPromoDayWeek.mean(),inplace=True)
dftest['MinSalesPromoDayWeeklog'] = dftest['MinSalesPromoDayWeek'].map(f)


data = None


#MaxSalesPromoDayWeek
data = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].max()
data = data.reset_index()
data.rename(columns={'Sales': 'MaxSalesPromoDayWeek'},inplace=True)
df = pd.merge(df,data, on = ['Store','DayOfWeek','Promo'],how ='left')
df.MaxSalesPromoDayWeek.fillna(data.MaxSalesPromoDayWeek.mean(),inplace=True)
df['MaxSalesPromoDayWeeklog'] = df['MaxSalesPromoDayWeek'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','DayOfWeek','Promo'],how="left")
dftest.MaxSalesPromoDayWeek.fillna(data.MaxSalesPromoDayWeek.mean(),inplace=True)
dftest['MaxSalesPromoDayWeeklog'] = dftest['MaxSalesPromoDayWeek'].map(f)

data = None

#MeanSalesPromoMonth
data = train.groupby(['Store','Month','Promo'])['Sales'].mean()
data = data.reset_index()
data.rename(columns={'Sales': 'MeanSalesPromoMonth'},inplace=True)
df = pd.merge(df,data, on = ['Store','Month','Promo'],how ='left')
df.MeanSalesPromoMonth.fillna(data.MeanSalesPromoMonth.mean(),inplace=True)
df['MeanSalesPromoMonthlog'] = df['MeanSalesPromoMonth'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','Month','Promo'],how="left")
dftest.MeanSalesPromoMonth.fillna(data.MeanSalesPromoMonth.mean(),inplace=True)
dftest['MeanSalesPromoMonthlog'] = dftest['MeanSalesPromoMonth'].map(f)


data = None



#MinSalesPromoMonth
data = train.groupby(['Store','Month','Promo'])['Sales'].min()
data = data.reset_index()
data.rename(columns={'Sales': 'MinSalesPromoMonth'},inplace=True)
df = pd.merge(df,data, on = ['Store','Month','Promo'], how ='left')
df.MinSalesPromoMonth.fillna(data.MinSalesPromoMonth.mean(),inplace=True)
df['MinSalesPromoMonthlog'] = df['MinSalesPromoMonth'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','Month','Promo'],how="left")
dftest.MinSalesPromoMonth.fillna(data.MinSalesPromoMonth.mean(),inplace=True)
dftest['MinSalesPromoMonthlog'] = dftest['MinSalesPromoMonth'].map(f)


data = None

#MaxSalesPromoMonth
data = train.groupby(['Store','Month','Promo'])['Sales'].max()
data = data.reset_index()
data.rename(columns={'Sales': 'MaxSalesPromoMonth'},inplace=True)
df = pd.merge(df,data, on = ['Store','Month','Promo'],how ='left')
df.MaxSalesPromoMonth.fillna(data.MaxSalesPromoMonth.mean(),inplace=True)
df['MaxSalesPromoMonthlog'] = df['MaxSalesPromoMonth'].map(f)

dftest = pd.merge(dftest,data, on = ['Store','Month','Promo'],how="left")
dftest.MaxSalesPromoMonth.fillna(data.MaxSalesPromoMonth.mean(),inplace=True)
dftest['MaxSalesPromoMonthlog'] = dftest['MaxSalesPromoMonth'].map(f)


data = None










#Export the data with the added features
df.to_csv("Data/train++.csv",index = False, header=True)
dftest.to_csv("Data/test++.csv",index = False, header=True)
