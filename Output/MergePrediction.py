import glob 
import pandas as pd 

files = glob.glob("predictions/*.csv")

df = pd.DataFrame(None,columns=['Id','Sales'])

for i in files:
    dft = pd.read_csv(i)
    df = df.append(dft)

df = df.sort('Id')

df['Id'] = df['Id'].astype(int) 
df.to_csv("predictions.csv",index=False,header=True)
