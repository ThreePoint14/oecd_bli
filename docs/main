import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

oecd = pd.read_csv(r'/Users/pariszhang/Desktop/Data/oecd/OECDBLI2017cleanedcsv.csv')

life_satisfaction = oecd['Life satisfaction as avg score']
personal_earnings = oecd['Personal earnings in usd']

d = {'Life Satisfaction': oecd['Life satisfaction as avg score'],
         'Personal Earnings': oecd['Personal earnings in usd']}

stats = pd.DataFrame(data=d)

x = np.array(stats['Life Satisfaction'])
y = np.array(stats['Personal Earnings'])

model = sklearn.linear_model.LinearRegression()
model.fit(x.reshape(-1, 1), y)

x_new = [[10]]
print(model.predict(x_new))
