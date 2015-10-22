

#This is a simple data exploration program that will produce 
#some figures I think might show us something interesting 
#and to investigate. 
#You will need pandas as well as seaborn

#to work with the data
import pandas as pd 
#to make pretty plot
import seaborn as sns
from datetime import datetime

sns.set(style="ticks", context="talk")


df = pd.read_csv("Data/train++.csv")



# g = sns.lmplot(x="NumbDays", y="Sales", data=df)

# sns.jointplot(kind= 'hex',x="NumbDays", y="Sales", data=df)

#Dist plot
sns.distplot(df['Sales'],kde=False, color="b")




#Sales for locations 
g = sns.boxplot(x = "DayOfWeek",y="Sales",data = df,order = [1,2,3,4,5,6,7],hue='StoreLoc')

x1,x2,y1,y2 = g.axis()

g.axis((x1,x2,0,40000))

#This is the day of the week sales. 
# sns.boxplot(x = "DayOfWeek",y="Sales",data = df,order = [1,2,3,4,5,6,7])



sns.plt.show()

# print df