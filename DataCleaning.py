

#This is the script used to extend the data set features


#to work with the data
import pandas as pd 
#to make pretty plot
import seaborn as sns
from datetime import datetime
import numpy as np



def UpdateDataSet(df,dfData=None):
    if dfData is None:
        dfData = df
    #only keep the data with positive values
    dfData = dfData[dfData['Sales']>0]


    #Add the mean of the month in right now. 
    fMonth = lambda x : values.loc[values['Month']== x].values[0][1]
    fDayWeek = lambda x : values.loc[values['DayOfWeek']== x].values[0][1]

    values =  (dfData.groupby(['Month'])['Sales'].mean()).reset_index()
    df['MeanMonthSale'] = dfData['Month'].map(fMonth)
    df['MeanMonthSale'].fillna(dfData.Sales.mean(),inplace=True)

    values =  (dfData.groupby(['DayOfWeek'])['Sales'].mean()).reset_index()
    df['MeanDayOfWeekSale'] = dfData['DayOfWeek'].map(fDayWeek)
    df['MeanDayOfWeekSale'].fillna(dfData.Sales.mean(),inplace=True)

    #Add the max sale of the month
    values =  (dfData.groupby(['Month'])['Sales'].max()).reset_index()
    df['MaxMonthSale'] = dfData['Month'].map(fMonth)
    df['MaxMonthSale'].fillna(dfData.Sales.max(),inplace=True)

    values =  (dfData.groupby(['DayOfWeek'])['Sales'].max()).reset_index()
    df['MaxDayOfWeekSale'] = dfData['DayOfWeek'].map(fDayWeek)
    df['MaxDayOfWeekSale'].fillna(dfData.Sales.max(),inplace=True)


    #add min sale of the months
    #Add the max sale of the month
    values =  (dfData.groupby(['Month'])['Sales'].min()).reset_index()
    df['MinMonthSale'] = dfData['Month'].map(fMonth)
    df['MinMonthSale'].fillna(dfData.Sales.min(),inplace=True)

    values =  (dfData.groupby(['DayOfWeek'])['Sales'].min()).reset_index()
    df['MinDayOfWeekSale'] = dfData['DayOfWeek'].map(fDayWeek)
    df['MinDayOfWeekSale'].fillna(dfData.Sales.min(),inplace=True)

    #Add the mean of people coming to the store on this month

    values =  (dfData.groupby(['Month'])['Customers'].mean()).reset_index()
    df['PeopleMeansMonth'] = dfData['Month'].map(fMonth)
    df['PeopleMeansMonth'].fillna(dfData.Sales.mean(),inplace=True)

    values =  (dfData.groupby(['DayOfWeek'])['Customers'].mean()).reset_index()
    df['PeopleMeansDayOfWeek'] = dfData['DayOfWeek'].map(fDayWeek)
    df['PeopleMeansDayOfWeek'].fillna(dfData.Sales.mean(),inplace=True)

    #add min and max of people
    values =  (dfData.groupby(['Month'])['Customers'].min()).reset_index()
    df['PeopleMinMonth'] = dfData['Month'].map(fMonth)
    df['PeopleMinMonth'].fillna(dfData.Sales.min(),inplace=True)

    values =  (dfData.groupby(['DayOfWeek'])['Customers'].min()).reset_index()
    df['PeopleMinDayOfWeek'] = dfData['DayOfWeek'].map(fDayWeek)
    df['PeopleMinDayOfWeek'].fillna(dfData.Sales.min(),inplace=True)


    values =  (dfData.groupby(['Month'])['Customers'].max()).reset_index()
    df['PeopleMaxMonth'] = dfData['Month'].map(fMonth)
    df['PeopleMaxMonth'].fillna(dfData.Sales.max(),inplace=True)

    values =  (dfData.groupby(['DayOfWeek'])['Customers'].max()).reset_index()
    df['PeopleMaxDayOfWeek'] = dfData['DayOfWeek'].map(fDayWeek)
    df['PeopleMaxDayOfWeek'].fillna(dfData.Sales.max(),inplace=True)


    return df




df = pd.read_csv("Data/train.csv")
#Some extra values
dfLoc = pd.read_csv("Data/store_states.csv")
dfStore = pd.read_csv('Data/store.csv')
dftest = pd.read_csv("Data/test.csv")




# df = df.loc[df['Store'] == storeID]

df = UpdateDataSet(df)

#Add certain feature as a function of the store id. 






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





f = lambda x : int(x[5:7])


#add the month in the feature: 
df['Month'] = df['Date'].map(f)





#Export the data with the added features
df.to_csv("Data/train++.csv",header=True)
