

#This is the script used to extend the data set features


#to work with the data
import pandas as pd 
#to make pretty plot
import seaborn as sns
from datetime import datetime


df = pd.read_csv("Data/train.csv")


#Add the column location
dfLoc = pd.read_csv("Data/store_states.csv")

print dfLoc

f = lambda x : dfLoc[dfLoc['Store']==x].values[0][1]

# print f(10)

# quit()

# df['StoreLoc'] = df['Store'].map(f)

#Convert the date to int of days. Year 2013-01-01 is the int 0. 
dayStart = datetime.strptime('2013-01-01',"%Y-%m-%d")

f = lambda x : (datetime.strptime(x,'%Y-%m-%d') - dayStart).days


df["NumbDays"] = df["Date"].map(f)

#Export the data with the added features
df.to_csv("Data/train++.csv",header=True)
