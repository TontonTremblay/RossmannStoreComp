#This script does a simple linear regression for 
#a single store. The test is using the month of July

import pandas as pd 
import seaborn as sns 
import numpy as np 
import datetime
import matplotlib.pyplot as plt
import os
#This is the ML part!!!
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


def main():
    #Set the colour and style
    sns.set(style="white", palette="muted", color_codes=True)
    pd.options.mode.chained_assignment = None

    #Test for one instance. 
    global testing
    testing = False


    df = pd.read_csv("Data/train++.csv")

    #feature list (NOT UDPATED)v
    #Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday,StoreLoc,NumbDays
    features = ['DayOfWeek','Open','Promo',"NumbDays",'Month','MeanMonthSale',\
    "MaxMonthSale","MinMonthSale",'PeopleMeansMonth','PeopleMinMonth','PeopleMaxMonth',\
    "PeopleMinDayOfWeek","PeopleMaxDayOfWeek","PeopleMeansDayOfWeek",\
    'MeanDayOfWeekSale','MinDayOfWeekSale',"MaxDayOfWeekSale",'SchoolHoliday',"StateHoliday"]



    if testing:
        #Only keep the one with specific id 
        #353 is pretty hard to fit with simple models. 
        #112 linear regression model is really good. 

        storeID = 579

        df = df.loc[df['Store'] == storeID]

        df = UpdateDataSet(df)

        startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
        endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - \
            datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days
        #Want to test on the last available month. This is closer 
        #to how this model is going to be test on the competition
        testData = df.loc[df['NumbDays']>=endDay]
        trainData = df.loc[df['NumbDays']<endDay]
        trainRegressorPredict(trainData,testData,features)

    else:
        #This is testing for everyone 
        testdf = pd.read_csv('Data/test.csv')

        #add numdays to dataset
        dayStart = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")

        f = lambda x : (datetime.datetime.strptime(x,'%Y-%m-%d') - dayStart).days

        testdf["NumbDays"] = testdf["Date"].map(f)

        #add the month to the DataSet
        f = lambda x : int(x[5:7])

        #add the month in the feature: 
        testdf['Month'] = testdf['Date'].map(f)
        
        # testdf = UpdateDataSet(testdf,df)

        #Test for each store and fill in the values. 
        storeIds = df['Store'].unique()
        
        for i in storeIds:
            r = []
            dft = df.loc[df['Store'] == i]
            dftTest = testdf.loc[df['Store'] == i] 
            dft = UpdateDataSet(dft)
            dftTest = UpdateDataSet(dftTest,dft)
            
            p = trainRegressorPredict(dft,dftTest,features)

            #Put the p at the right place with the right id. 
            c = 0
            for j in dftTest['Id']:
                r.append([j,p[c]])
                c+=1
            
            # break
            r = sorted(r)
            r = pd.DataFrame(np.array(r),columns=['Id','Sales'])
            r.to_csv("Output/predictions/predict"+str(i)+".csv",index=False,header=True)
            break
        #Save the output



def UpdateDataSet(df,dfData=None):
    if dfData is None:
        dfData = df
    #Add the mean of the month in right now. 
    fMonth = lambda x : values.loc[values['Month']== x].values[0][1]
    fDayWeek = lambda x : values.loc[values['DayOfWeek']== x].values[0][1]

    values =  (dfData.groupby(['Month'])['Sales'].mean()).reset_index()
    df['MeanMonthSale'] = dfData['Month'].map(fMonth)

    values =  (dfData.groupby(['DayOfWeek'])['Sales'].mean()).reset_index()
    df['MeanDayOfWeekSale'] = dfData['DayOfWeek'].map(fDayWeek)

    #Add the max sale of the month
    values =  (dfData.groupby(['Month'])['Sales'].max()).reset_index()
    df['MaxMonthSale'] = dfData['Month'].map(fMonth)

    values =  (dfData.groupby(['DayOfWeek'])['Sales'].max()).reset_index()
    df['MaxDayOfWeekSale'] = dfData['DayOfWeek'].map(fDayWeek)

    #add min sale of the months
    #Add the max sale of the month
    values =  (dfData.groupby(['Month'])['Sales'].min()).reset_index()
    df['MinMonthSale'] = dfData['Month'].map(fMonth)

    values =  (dfData.groupby(['DayOfWeek'])['Sales'].min()).reset_index()
    df['MinDayOfWeekSale'] = dfData['DayOfWeek'].map(fDayWeek)

    #Add the mean of people coming to the store on this month

    values =  (dfData.groupby(['Month'])['Customers'].mean()).reset_index()
    df['PeopleMeansMonth'] = dfData['Month'].map(fMonth)

    values =  (dfData.groupby(['DayOfWeek'])['Customers'].mean()).reset_index()
    df['PeopleMeansDayOfWeek'] = dfData['DayOfWeek'].map(fDayWeek)

    #add min and max of people
    values =  (dfData.groupby(['Month'])['Customers'].min()).reset_index()
    df['PeopleMinMonth'] = dfData['Month'].map(fMonth)

    values =  (dfData.groupby(['DayOfWeek'])['Customers'].min()).reset_index()
    df['PeopleMinDayOfWeek'] = dfData['DayOfWeek'].map(fDayWeek)


    values =  (dfData.groupby(['Month'])['Customers'].max()).reset_index()
    df['PeopleMaxMonth'] = dfData['Month'].map(fMonth)

    values =  (dfData.groupby(['DayOfWeek'])['Customers'].max()).reset_index()
    df['PeopleMaxDayOfWeek'] = dfData['DayOfWeek'].map(fDayWeek)

    return df




def trainRegressorPredict(trainData,testData,features):
    global testing 
    Xtrain = trainData[features].values
    ytrain = trainData['Sales'].values

    xtest = testData[features].values
    global testing 
    if testing:
        ytest = testData['Sales'].values





    # clf = linear_model.Lasso(alpha = 1)
    clf = linear_model.LinearRegression()
    clf = RandomForestRegressor()
    clf.fit(Xtrain,ytrain)

    p = clf.predict(xtest)



    if testing:
        score = clf.score(xtest,ytest)

        for i in range( len (clf.feature_importances_)):
            print features[i], ':', clf.feature_importances_[i]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.scatter(testData['NumbDays'].values,p,color='b')
        plt.plot(testData['NumbDays'].values,p,color='b',alpha = 0.9,label='pre')

        plt.scatter(testData['NumbDays'].values,ytest,color='g')
        plt.plot(testData['NumbDays'].values,ytest,color='g',alpha = 0.9,label='real')
        # plt.label()

        x1,x2,y1,y2 = plt.axis()

        startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
        endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - \
            datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days



        plt.axis((endDay-1,x2,0,y2))
        plt.title('id: ' + str(trainData['Store'].unique()[0]) +  ' score: '+ str(score))

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
    return p






if __name__ == "__main__":
    main()




