
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
from scipy.stats import norm 
from sklearn.preprocessing import StandardScaler 
from scipy import stats 
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
print(df_train.columns)
print(df_train['SalePrice'].describe())

histogram = sns.distplot(df_train['SalePrice']);
fig1 = histogram.get_figure()
fig1.savefig('fig1.png')

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
scatterfig = data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
fig2 = scatterfig.get_figure()
fig2.savefig('fig2.png')

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
scatterfig2 = data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));
fig3 = scatterfig2.get_figure()
fig3.savefig('fig3.png')


var = 'OverallQual'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
f , ax = plt.subplots(figsize=(8,6))
boxplot = sns.boxplot(x=var,y='SalePrice',data=data)
boxplot.axis(ymin=0,ymax=800000);
fig4 = boxplot.get_figure()
fig4.savefig('fig4.png')

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
f,ax = plt.subplots(figsize=(16,8))
boxplot2 = sns.boxplot(x=var,y='SalePrice',data=data)
boxplot2.axis(ymin=0,ymax=800000);
plt.xticks(rotation=90);
fig5 = boxplot2.get_figure()
fig5.savefig('fig5.png')

corrmat = df_train.corr()
f,ax = plt.subplots(figsize=(12,9))
heatmap = sns.heatmap(corrmat,vmax=0.8,square=True);
fig6 = heatmap.get_figure()
fig6.savefig('fig6.png')

k = 10 
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True, fmt='.2f', annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
#plt.show()
fig7 = hm.get_figure()
fig7.savefig('fig7.png')
#plt.savefig('fig7.png')


sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(df_train[cols],size=2.5)
#plt.show()
plt.savefig('fig8.png')

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
print(missing_data.head(20))

df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
scatter9 = data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));
fig9 = scatter9.get_figure()
fig9.savefig('fig9.png')


df_train.sort_values(by='GrLivArea',ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id']==524].index)

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
scatter10 = data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));
fig10 = scatter10.get_figure()
fig10.savefig('fig10.png')


sns.distplot(df_train['SalePrice'],fit=norm);
fig11 = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot=plt)
fig11.savefig('fig11.png')
#fig12 = plt.figure()
#fig12.savefig('fig12.png')

df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'],fit=norm);
fig12 = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot=plt)
fig12.savefig('fig12.png')


dist13 = sns.distplot(df_train['GrLivArea'],fit=norm);
fig13 = dist13.get_figure()
fig13.savefig('fig13.png')

fig14 = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot=plt)
fig14.savefig('fig14.png')

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

dist15 = sns.distplot(df_train['GrLivArea'],fit=norm);
#fig15 = plt.figure()
fig15 = dist15.get_figure()
fig15.savefig('fig15.png')

res = stats.probplot(df_train['GrLivArea'],plot=plt)
#fig16 = res.get_figure()
plt.savefig('fig16.png')
#fig16.savefig('fig16.png')

