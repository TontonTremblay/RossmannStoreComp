#This script does a simple linear regression for 
#a single store. The test is using the month of July

#This approach scored: 0.66700 :S not better than simply using 
#the median

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
import sklearn
import math

def main():
    #Set the colour and style
    sns.set(style="white", palette="muted", color_codes=True)
    pd.options.mode.chained_assignment = None

    #Test for one instance. 
    global testing,contestOutput
    contestOutput = False
    testing = True


    df = pd.read_csv("Data/train++.csv")

    #feature list (NOT UDPATED)v

    '''
    Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,
    SchoolHoliday,NumbDays,Month,Saleslog,MeanSalesDayOfWeek,
    MeanSalesDayOfWeeklog,MinSalesDayWeek,MinSalesDayWeeklog,
    MaxSalesDayWeek,MaxSalesDayWeeklog,MeanSalesPromoDayWeek,
    MeanSalesPromoDayWeeklog,MinSalesPromoDayWeek,MinSalesPromoDayWeeklog,
    MaxSalesPromoDayWeek,MaxSalesPromoDayWeeklog,MeanSalesPromoMonth,
    MeanSalesPromoMonthlog,MinSalesPromoMonth,MinSalesPromoMonthlog,
    MaxSalesPromoMonth,MaxSalesPromoMonthlog
    '''
    features = ['DayOfWeek','Open','Promo',"NumbDays",'Month',\
        'SchoolHoliday',"StateHoliday","MeanSalesDayOfWeeklog",
        "MinSalesDayWeeklog","MaxSalesDayWeeklog","MeanSalesPromoDayWeeklog",
        "MinSalesPromoDayWeeklog", "MaxSalesPromoDayWeeklog",
        "MeanSalesPromoMonthlog","MinSalesPromoMonthlog",
        "MaxSalesPromoMonthlog"]



    # features = ['DayOfWeek','Open','Promo',"NumbDays",'Month',\
    #     'SchoolHoliday',"StateHoliday","MeanSalesDayOfWeek",
    #     "MinSalesDayWeek","MaxSalesDayWeek","MeanSalesPromoDayWeek",
    #     "MinSalesPromoDayWeek", "MaxSalesPromoDayWeek",
    #     "MeanSalesPromoMonth","MinSalesPromoMonth",
    #     "MaxSalesPromoMonth"]


    if testing:
        #Only keep the one with specific id 
        #353 is pretty hard to fit with simple models. 
        #112 linear regression model is really good. 

        #prior analysis to check: 292

        storeID = 123

        df = df.loc[df['Store'] == storeID]


        startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
        endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - \
            datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days
        #Want to test on the last available month. This is closer 
        #to how this model is going to be test on the competition
        testData = df.loc[df['NumbDays']>=endDay]
        trainData = df.loc[df['NumbDays']<endDay]
        trainRegressorPredict(trainData,testData,features)

    else:
        if contestOutput:
            #This is testing for everyone 
            testdf = pd.read_csv('Data/test++.csv')

            #add numdays to dataset


            #Test for each store and fill in the values. 
            storeIds = df['Store'].unique()
            
            for i in range(0,len(storeIds)):
                i = storeIds[i]
                r = []
                dft = df.loc[df['Store'] == i]
                dftTest = testdf.loc[df['Store'] == i] 
                # print dftTest
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
                # print i,
                # break
        else:
            startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
            endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - \
                datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days
            #Want to test on the last available month. This is closer 
            #to how this model is going to be test on the competition
            testData = df.loc[df['NumbDays']>=endDay]
            trainData = df.loc[df['NumbDays']<endDay]

            storeIds = df['Store'].unique()
            results = []
            for i in range(1,len(storeIds)):
                dft = trainData.loc[trainData['Store'] == i]
                dftTest = testData.loc[testData['Store'] == i] 

                results.append([i,ErrorPerStore(dft,dftTest,features)])
            # print np.array(results)
            output = pd.DataFrame(np.array(results),columns=['Store','Score'])
            print output.Score.describe()
            output.to_csv('Output/julyScore.csv',index=False,header=True)
            


def fmap(x):
    if x is 0:
        return 0
    else:
        return 1 



def trainRegressorPredict(trainData,testData,features):
    global testing 
    Xtrain = trainData[features].values
    ytrain = trainData['Sales'].values

    xtest = testData[features].values
    global testing 
    if testing:
        ytest = testData['Sales'].values





    # clf = linear_model.Lasso(alpha = 50) 
    # clf = linear_model.LinearRegression()
    clf = RandomForestRegressor()
    # clf = sklearn.svm.LinearSVR(C=15)
    clf.fit(Xtrain,ytrain)

    p = clf.predict(xtest)

    def RootMeanSquaredError(ytest,y):
        s = 0
        for i in range(len(y)):
            yi = ytest[i]
            yh = y[i]
            if yi >0:
                v = (yi-yh)/yi
                v*=v
                s+=v
        return math.sqrt( (1.0/len(ytest))*s)

    if testing:
        #plot and check what is going on!?!?!?

        

        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        #Simple predictor
        train = trainData.loc[trainData['Sales']>0]

        meansStores = train.groupby(['DayOfWeek','Promo'])['Sales'].mean()
        meansStores = meansStores.reset_index()

        resMean = pd.merge(testData,meansStores, on = ['DayOfWeek','Promo'],how ='left')
        resMean.fillna(train.Sales.mean(),inplace=True)

        plt.scatter(testData['NumbDays'].values,resMean['Sales_y'],color='y',alpha=0.8)
        plt.plot(testData['NumbDays'].values,resMean['Sales_y'],color='y',alpha=0.8,\
            label='mean:' + \
            str("%.3f" % RootMeanSquaredError(ytest,resMean['Sales_y'].values)))

        plt.scatter(testData['NumbDays'].values,p,color='b')
        plt.plot(testData['NumbDays'].values,p,color='b',alpha = 0.9,\
            label='prec:' + \
            str("%.3f" % RootMeanSquaredError(ytest,p)))

        plt.scatter(testData['NumbDays'].values,ytest,color='g')
        plt.plot(testData['NumbDays'].values,ytest,color='g',alpha = 0.9,label='real')
        # plt.label()

        x1,x2,y1,y2 = plt.axis()

        startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
        endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - \
            datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days



        plt.axis((endDay-1,x2,0,y2))
        plt.title('id: ' + str(trainData['Store'].unique()[0]))

        #Add the legend as well 
        plt.legend(loc=4)

        # check out how to change for the dates. 
        values = ax.xaxis.get_majorticklocs()
        labels2 = []
        for v in values:
            end_date = startDay + datetime.timedelta(days=v)
            labels2.append(end_date.strftime("%d-%m-%y"))
        
        ax.set_xticklabels(labels2)

        # print some values/debugs
        for i in range( len (clf.feature_importances_)):
            print features[i], ':', clf.feature_importances_[i]


        print ""
        print "mean predictor:", str("%.5f" % RootMeanSquaredError(ytest,resMean['Sales_y'].values))
        print "test predictor:", str("%.5f" % RootMeanSquaredError(ytest,p))
        plt.show()



    return p

def ErrorPerStore(trainData,testData,features):

    Xtrain = trainData[features].values
    ytrain = trainData['Sales'].values

    xtest = testData[features].values

    ytest = testData['Sales'].values





    # clf = linear_model.Lasso(alpha = 50) 
    # clf = linear_model.LinearRegression()
    clf = RandomForestRegressor()
    # clf = sklearn.svm.LinearSVR(C=15)
    clf.fit(Xtrain,ytrain)

    p = clf.predict(xtest)

    return RootMeanSquaredError(ytest,p)


def RootMeanSquaredError(ytest,y):
    s = 0
    for i in range(len(y)):
        yi = ytest[i]
        yh = y[i]
        if yi >0:
            v = (yi-yh)/yi
            v*=v
            s+=v
    return math.sqrt( (1.0/len(ytest))*s)
if __name__ == "__main__":
    main()




