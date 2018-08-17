# -*- coding:utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('Bike_Train.csv')
test = pd.read_csv('Bike_Test.csv')

print("original train data:", train.shape)
print("original test data:", test.shape)

#图1：count
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.set_size_inches(6,5)
sns.distplot(train['count'])
ax.set(xlabel='count', title='distribution')
fig.savefig('001 count distribution', dpi=200)

train_drop_tail = train[np.abs(train['count']- train['count'].mean())<=(3*train['count'].std())]
print('----------------------')
print("train_drop_tail:", train_drop_tail.shape)

#图2 两个图的对比
fig = plt.figure()
fig.set_size_inches(12,5)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.distplot(train['count'], ax=ax1)
sns.distplot(train_drop_tail['count'], ax=ax2)
ax1.set(xlabel='count', title='distribution former')
ax2.set(xlabel='former count', title='distribution after')
fig.savefig('002 count distribution compare', dpi=200)

fd = pd.concat([train_drop_tail, test], ignore_index=True)
print(fd.shape)

#自己写的转换时间格式
fd['datetime'] = pd.to_datetime(fd['datetime'], format='%d/%m/%Y %H:%M')
fd['year'] = fd['datetime'].dt.year
fd['month'] = fd['datetime'].dt.month
fd['day'] = fd['datetime'].dt.day
fd['hour'] = fd['datetime'].dt.hour
fd['minute'] = fd['datetime'].dt.minute

print('--------------------\n check for date transfer:')
print(fd.columns)
print(fd.dtypes)
fd.to_csv('fd.csv')

#图3 画四个对比图
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(12,10)

sns.distplot(fd['temp'],ax=axes[0,0])
sns.distplot(fd['atemp'],ax=axes[0,1])
sns.distplot(fd['humidity'],ax=axes[1,0])
sns.distplot(fd['windspeed'],ax=axes[1,1])

axes[0,0].set(xlabel='temp',title='Distribution of temp',)
axes[0,1].set(xlabel='atemp',title='Distribution of atemp')
axes[1,0].set(xlabel='humidity',title='Distribution of humidity')
axes[1,1].set(xlabel='windspeed',title='Distribution of windspeed')

fig.savefig('003 four features comparision', dpi=200)

#图4:时段对租赁影响
workingday_fd = fd[fd['workingday'] == 1]
workingday_fd = workingday_fd.groupby(['hour'], as_index = True).agg({'casual':'mean', 'registered':'mean', 'count':'mean'})

noworking_fd = fd[fd['workingday'] == 0]
noworking_fd = noworking_fd.groupby(['hour'], as_index = True).agg({'casual':'mean', 'registered':'mean', 'count':'mean'})

fig, axes = plt.subplots(1,2,sharey= True)
workingday_fd.plot(figsize=(15,5),title = 'Working day rentails', ax = axes[0])
noworking_fd.plot(figsize=(15,5),title = 'Noneworking day retails', ax = axes[1])
fig.savefig('004 days and rentails', dpi=200)



#005:月份和租赁

sns.boxplot(fd['month'], fd['count'])
fig.savefig('005 months and rentails', dpi=200)


#006 湿度和租赁
humidity_df = fd.groupby('datetime', as_index=False).agg({'humidity':'mean'})
humidity_df['datetime']=pd.to_datetime(humidity_df['datetime'])
#将日期设置为时间索引
humidity_df=humidity_df.set_index('datetime')

humidity_month = fd.groupby(['year','month'], as_index=False).agg({'day':'min','humidity':'mean'})
humidity_month.rename(columns={'day':'day'},inplace=True)
humidity_month['datetime']=pd.to_datetime(humidity_month[['year','month','day']])

fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(1,1,1)
plt.plot(humidity_df.index , humidity_df['humidity'] , linewidth=1.3,label='Daily average')
plt.plot(humidity_month['datetime'], humidity_month['humidity'] ,marker='o', linewidth=1.3,label='Monthly average')
ax.legend()
ax.set_title('Change trend of average humidity per day in two years')
fig.savefig('006 humidity and rentails', dpi=200)

#007年和租赁
sns.boxplot(fd['year'], fd['count'])
fig.savefig('007 year and rentails', dpi=200)

#008 季节和租赁
day_df=fd.groupby('datetime').agg({'year':'mean','season':'mean','casual':'sum', 'registered':'sum','count':'sum','temp':'mean', 'atemp':'mean'})
season_df = day_df.groupby(['year','season'], as_index=True).agg({'casual':'mean', 'registered':'mean', 'count':'mean'})
temp_df = day_df.groupby(['year','season'], as_index=True).agg({'temp':'mean', 'atemp':'mean'})
fig.savefig('008 season and rentails', dpi=200)

#009 and 010 天气和租赁
sns.boxplot(fd['weather'], fd['count'])
fig.savefig('009 weather and rentails', dpi=200)
sns.pointplot(x='hour', y='count', hue='weather', data=fd)
fig.savefig('010 weather and rentails sort by hour', dpi=200)





dummies_month = pd.get_dummies(fd['month'], prefix= 'month')
dummies_season=pd.get_dummies(fd['season'],prefix='season')
dummies_weather=pd.get_dummies(fd['weather'],prefix='weather')
dummies_year=pd.get_dummies(fd['year'],prefix='year')
fd=pd.concat([fd,dummies_month,dummies_season,dummies_weather,dummies_year],axis=1)

dataTrain = fd[pd.notnull(fd['count'])]
dataTest= fd[~pd.notnull(fd['count'])].sort_values(by=['datetime'])
datetimecol = dataTest['datetime']
yLabels=dataTrain['count']
yLabels_log=np.log(yLabels)

dropFeatures = ['casual' , 'count' , 'datetime' , 'day' , 'registered' ,'windspeed' , 'atemp' , 'month','season','weather', 'year' ]
dataTrain = dataTrain.drop(dropFeatures , axis=1)
dataTest = dataTest.drop(dropFeatures , axis=1)

rfModel = RandomForestRegressor(n_estimators=1000 , random_state = 42)
rfModel.fit(dataTrain , yLabels_log)
preds = rfModel.predict( X = dataTrain)


predsTest= rfModel.predict(X = dataTest)
submission=pd.DataFrame({'datetime':datetimecol , 'count':[max(0,x) for x in np.exp(predsTest)]})
submission.to_csv('bike_predictions.csv',index=False)

print(rfModel.score(dataTrain , yLabels_log))

