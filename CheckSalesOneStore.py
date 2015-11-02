#to work with the data
import pandas as pd 
#to make pretty plot
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

sns.set(style="white", palette="muted", color_codes=True)

idStore = 292

df = pd.read_csv("Data/train++.csv")
df = df[df['Store']==idStore]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(df['NumbDays'],df['Sales'],color='g')
x1,x2,y1,y2 = plt.axis()
x1 = -5
x2 = df['NumbDays'].count()
y1 = 0 
plt.axis([x1,x2,y1,y2])


startDay = datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")
endDay = (datetime.datetime.strptime('2015-07-01',"%Y-%m-%d") - \
    datetime.datetime.strptime('2013-01-01',"%Y-%m-%d")).days
values = ax.xaxis.get_majorticklocs()
labels2 = []
for v in values:
    end_date = startDay + datetime.timedelta(days=v)
    labels2.append(end_date.strftime("%d-%m-%y"))

ax.set_xticklabels(labels2)

plt.show()
plt.clf()