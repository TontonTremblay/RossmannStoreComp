#This script does a simple linear regression for 
#a single store. The test is using the month of July

import pandas as pd 
import seaborn as sns 
import numpy as np 
import datetime
import matplotlib.pyplot as plt
import os

#Set the colour and style
sns.set(style="white", palette="muted", color_codes=True)


df = pd.read_csv("Data/train++.csv")

#Only keep the one with specific id 
#353 is pretty hard to fit with simple models. 
#112 linear regression model is really good. 

storeID = 112

df = df.loc[df['Store'] == storeID]


#Find the test data set. 2013/01/01 is the day 0 
startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days


#Want to test on the last available month. This is closer 
#to how this model is going to be test on the competition
testData = df.loc[df['NumbDays']>=endDay]
trainData = df.loc[df['NumbDays']<endDay]

#Features available:
#Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday,StoreLoc,NumbDays

features = ['DayOfWeek','Customers','Open','Promo',"NumbDays",'Month']


Xtrain = trainData[features].values
ytrain = trainData['Sales'].values

xtest = testData[features].values
ytest = testData['Sales'].values



#This is the ML part!!!
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


clf = linear_model.LinearRegression()
# clf = RandomForestClassifier()
clf.fit(Xtrain,ytrain)

p = clf.predict(xtest)

score = clf.score(xtest,ytest)
print score

#Produces a couple figures to save to Output/+idstore

directory = 'Output/'+str(storeID)+'/'

#Check if the folder exists
if not os.path.exists(directory):
    os.makedirs(directory)

#Check how the predictor performs using a nice plot. 

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(testData['NumbDays'].values,p,color='b')
plt.plot(testData['NumbDays'].values,p,color='b',alpha = 0.9,label='predict')

plt.scatter(testData['NumbDays'].values,ytest,color='g')
plt.plot(testData['NumbDays'].values,ytest,color='g',alpha = 0.9,label='real')
# plt.label()

x1,x2,y1,y2 = plt.axis()




plt.axis((endDay-1,x2,0,y2))
plt.title('id: ' + str(storeID) +  ' score: '+ str(score))

#Add the legend as well 
plt.legend()

# check out how to change for the dates. 
values = ax.xaxis.get_majorticklocs()
labels2 = []
for v in values:
    end_date = startDay + datetime.timedelta(days=v)
    labels2.append(end_date.strftime("%d-%m-%y"))
    
ax.set_xticklabels(labels2)

plt.show()
quit()
# plt.savefig(directory+"performance.png")
plt.clf()


#Producing a plot of the sales over the data set.  
#Find the data points with 
dft = df.loc[df['Sales']>0]
ax = sns.lmplot(x="NumbDays", y="Sales",data = dft, size = 9,order=1).ax
dft = df.loc[df['Promo']==1]

plt.scatter(dft['NumbDays'].values,dft['Sales'].values,color='g',s=25,marker='s')


x1,x2,y1,y2 = plt.axis()
plt.axis((0,x2,-1,max(df['Sales'])))

values = ax.xaxis.get_majorticklocs()
labels2 = []
for v in values:
    end_date = startDay + datetime.timedelta(days=v)
    labels2.append(end_date.strftime("%d-%m-%y"))
    
ax.set_xticklabels(labels2)


plt.savefig(directory+"SalesData.png")

plt.clf()

#plot of the sales as a function of the day of the week
g = sns.boxplot(x = "DayOfWeek",y="Sales",data = df,order = [1,2,3,4,5,6,7])

x1,x2,y1,y2 = g.axis()

g.axis((x1,x2,0,max(df['Sales'])))
# plt.show()
plt.savefig(directory+"SalesDayWeek.png")

