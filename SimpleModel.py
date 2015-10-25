#predict as historical mean given store, day_week and promo
import pandas as pd 
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")


train = train.loc[train['Sales']>0]

meansStores = train.groupby(['Store','DayOfWeek','Promo'])['Sales'].mean()
meansStores = meansStores.reset_index()

# print meansStores

test = pd.merge(test,meansStores, on = ['Store','DayOfWeek','Promo'],how ='left')
test.fillna(train.Sales.mean(),inplace=True)


test[['Id','Sales']].to_csv('Output/base_prediction.csv',index=False)
