

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

# df['StoreLoc'] = df['Store'].map(f)


#Convert the a to 1
def f(x):
	if x is 0:
		return 0
	else:
		return 1 

df['StateHoliday'] = df['StateHoliday'].map(f)


#Convert the date to int of days. Year 2013-01-01 is the int 0. 
dayStart = datetime.strptime('2013-01-01',"%Y-%m-%d")
f = lambda x : (datetime.strptime(x,'%Y-%m-%d') - dayStart).days
df["NumbDays"] = df["Date"].map(f)

#add the month in the feature: 
f = lambda x : int(x[5:7])
df['Month'] = df['Date'].map(f)

#Add log sales values
f = lambda x : np.log(x+1)
df['Saleslog'] = df['Sales'].map(f)



train = df.loc[df['Sales']>0]
#Add some simple features e.g. 
#MeanSalesDayWeek
meansSalesStores = train.groupby(['Store','DayOfWeek'])['Sales'].mean()
meansSalesStores = meansSalesStores.reset_index()
meansSalesStores.rename(columns={'Sales': 'MeanSalesDayOfWeek'},inplace=True)
df = pd.merge(df,meansSalesStores, on = ['Store','DayOfWeek'],left_index=True, right_index=True,how ='outer')
df.MeanSalesDayOfWeek.fillna(meansSalesStores.MeanSalesDayOfWeek.mean(),inplace=True)
df['MeanSalesDayOfWeeklog'] = df['MeanSalesDayOfWeek'].map(f)
meansSalesStores = None


#MinSalesDayWeek
minsSalesStores = train.groupby(['Store','DayOfWeek'])['Sales'].min()
minsSalesStores = minsSalesStores.reset_index()
minsSalesStores.rename(columns={'Sales': 'MinSalesDayWeek'},inplace=True)
df = pd.merge(df,minsSalesStores, on = ['Store','DayOfWeek'], how ='left')
df.MinSalesDayWeek.fillna(minsSalesStores.MinSalesDayWeek.mean(),inplace=True)
df['MinSalesDayWeeklog'] = df['MinSalesDayWeek'].map(f)
minsSalesStores = None

#MaxSalesDayWeek
maxsSalesStores = train.groupby(['Store','DayOfWeek'])['Sales'].max()
maxsSalesStores = maxsSalesStores.reset_index()
maxsSalesStores.rename(columns={'Sales': 'MaxSalesDayWeek'},inplace=True)
df = pd.merge(df,maxsSalesStores, on = ['Store','DayOfWeek'],how ='left')
df.MaxSalesDayWeek.fillna(maxsSalesStores.MaxSalesDayWeek.mean(),inplace=True)
df['MaxSalesDayWeeklog'] = df['MaxSalesDayWeek'].map(f)
maxsSalesStores = None

#MeanSalesPromoDayWeek
meansSalesStores = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].mean()
meansSalesStores = meansSalesStores.reset_index()
meansSalesStores.rename(columns={'Sales': 'MeanSalesPromoDayWeek'},inplace=True)
df = pd.merge(df,meansSalesStores, on = ['Store','DayOfWeek','Promo'],how ='left')
df.MeanSalesPromoDayWeek.fillna(meansSalesStores.MeanSalesPromoDayWeek.mean(),inplace=True)
df['MeanSalesPromoDayWeeklog'] = df['MeanSalesPromoDayWeek'].map(f)
meansSalesStores = None

#MinSalesPromoDayWeek
minsSalesStores = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].min()
minsSalesStores = minsSalesStores.reset_index()
minsSalesStores.rename(columns={'Sales': 'MinSalesPromoDayWeek'},inplace=True)
df = pd.merge(df,minsSalesStores, on = ['Store','DayOfWeek','Promo'], how ='left')
df.MinSalesPromoDayWeek.fillna(minsSalesStores.MinSalesPromoDayWeek.mean(),inplace=True)
df['MinSalesPromoDayWeeklog'] = df['MinSalesPromoDayWeek'].map(f)
minsSalesStores = None


#MaxSalesPromoDayWeek
maxsSalesStores = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].max()
maxsSalesStores = maxsSalesStores.reset_index()
maxsSalesStores.rename(columns={'Sales': 'MaxSalesPromoDayWeek'},inplace=True)
df = pd.merge(df,maxsSalesStores, on = ['Store','DayOfWeek','Promo'],how ='left')
df.MaxSalesPromoDayWeek.fillna(maxsSalesStores.MaxSalesPromoDayWeek.mean(),inplace=True)
df['MaxSalesPromoDayWeeklog'] = df['MaxSalesPromoDayWeek'].map(f)
maxsSalesStores = None

#MeanSalesPromoMonth
meansSalesStores = train.groupby(['Store','Month','Promo'])['Sales'].mean()
meansSalesStores = meansSalesStores.reset_index()
meansSalesStores.rename(columns={'Sales': 'MeanSalesPromoMonth'},inplace=True)
df = pd.merge(df,meansSalesStores, on = ['Store','Month','Promo'],how ='left')
df.MeanSalesPromoMonth.fillna(meansSalesStores.MeanSalesPromoMonth.mean(),inplace=True)
df['MeanSalesPromoMonthlog'] = df['MeanSalesPromoMonth'].map(f)
meansSalesStores = None

#MinSalesPromoMonth
minsSalesStores = train.groupby(['Store','Month','Promo'])['Sales'].min()
minsSalesStores = minsSalesStores.reset_index()
minsSalesStores.rename(columns={'Sales': 'MinSalesPromoMonth'},inplace=True)
df = pd.merge(df,minsSalesStores, on = ['Store','Month','Promo'], how ='left')
df.MinSalesPromoMonth.fillna(minsSalesStores.MinSalesPromoMonth.mean(),inplace=True)
df['MinSalesPromoMonthlog'] = df['MinSalesPromoMonth'].map(f)
minsSalesStores = None

#MaxSalesPromoMonth
maxsSalesStores = train.groupby(['Store','Month','Promo'])['Sales'].max()
maxsSalesStores = maxsSalesStores.reset_index()
maxsSalesStores.rename(columns={'Sales': 'MaxSalesPromoMonth'},inplace=True)
df = pd.merge(df,maxsSalesStores, on = ['Store','Month','Promo'],how ='left')
df.MaxSalesPromoMonth.fillna(maxsSalesStores.MaxSalesPromoMonth.mean(),inplace=True)
df['MaxSalesPromoMonthlog'] = df['MaxSalesPromoMonth'].map(f)
maxsSalesStores = None










#Export the data with the added features
df.to_csv("Data/train++.csv",index = False, header=True)
