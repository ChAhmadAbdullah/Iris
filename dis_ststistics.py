import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import parallel_coordinates
import plotly.express as px
from scipy import stats as st


iris_df = pd.read_csv('iris_csv.csv')

def mean_fun(label):
    data = iris_df[label]
    return (np.mean(data))

def mode_fun(label):
    return (iris_df[label].mode())

def median_fun(label):
    data = iris_df[label]
    return (np.median(data))


def varience_fun(label):
    data = iris_df[label]
    return (np.var(data))


def standerdeviation_fun(label):
    data = iris_df[label]
    return (np.std(data))


def max_fun(label):
    data = iris_df[label]
    return (np.max(data))


def min_fun(label):
    data = iris_df[label]
    return (np.min(data))

def quartile_fun(label):
    data= iris_df[label]
    return (np.percentile(data,q=25))

def quartile2_fun(label):
    data= iris_df[label]
    return (np.percentile(data,q=75))



def range_fun(label):
    return (max_fun(label) - min_fun(label))
print(iris_df.keys())

def correlation(label1, label2):
    return (iris_df[label1].corr(iris_df[label2]))

def correl_formula(label1, label2):
    mean_label1 = mean_fun(label1)
    mean_label2 = mean_fun(label2)
    b = len(iris_df[label1])
    print(b)
    var3 = 0
    var4 = 0
    var5 = 0
    for i in range(0, b):
        var = iris_df.at[i, label1]
        var2 = iris_df.at[i, label2]
        var3 += (mean_label1 - var) * (mean_label2 - var2)
    for i in iris_df[label1]:
        var4 += (mean_label1 - i)**2
    for i in iris_df[label2]:
        var5 += (mean_label2 - i)**2
        var6 = np.sqrt(var4*var5)
    return (var3/var6)


# arr = []
# for i in iris_df['sepallength']:
#     arr.append(i)
# print(st.mode(arr))
#
# print('mode =', st.mode(iris_df['sepallength']))
# print(iris_df.mode(axis='column',numeric_only=False, dropna= True))
# print(np.mod(iris_df['sepallength'],axis=0, numeric_only=False, dropna=True))
# single value

print('Mean = ', mean_fun('sepallength'))
print('Mode = ', mode_fun('sepallength'))
print('Min value =', min_fun('sepalwidth'))
print('Max value =', max_fun('sepalwidth'))
print('Range =', range_fun('sepalwidth'))

print(iris_df.describe())
print(iris_df.info())

# multivariate
# mean
arr = []
for i in iris_df.keys():
    try:
        arr.append(mean_fun(i))
    except:
        pass
print('Multivartiate Mean= ',np.mean(arr))

# medain
arr = []
for i in iris_df.keys():
    try:
        arr.append(median_fun(i))
    except:
        pass
print('Multivartiate Mode = ',np.median(arr))

# varience
arr = []
for i in iris_df.keys():
    try:
        arr.append(varience_fun(i))
    except:
        pass

print('Multivartiate Varience = ',np.var(arr))

# stander deviation
arr = []
for i in iris_df.keys():
    try:
        arr.append(standerdeviation_fun(i))
    except:
        pass
print('Multivartiate Stander Deviation =', np.std(arr))

# max value
arr = []
for i in iris_df.keys():
    try:
        arr.append(max_fun(i))
    except:
        pass
arr.pop(-1)
max_value = max(arr)
print('Multivariate Max_value =', max_value)
# min value
arr = []
for i in iris_df.keys():
    try:
        arr.append(min_fun(i))
    except:
        pass
arr.pop(-1)
min_value = min(arr)
print(min_value)

# range
print('Multivariate Range = ', max_value - min_value)

# mode
arr = []
for i in iris_df.keys():
    arr.append(mode_fun(i))
print(arr)

# Co-relation
print('correlation: ',correlation('sepallength', 'petallength'))
print(iris_df.corr())

#co-relation by formula
print('correlation by self made formula: ', correl_formula('sepallength', 'petallength'))


#Corelation plot
plt.scatter(iris_df['sepallength'], iris_df['petallength'],color='grey')
plt.plot(np.unique(iris_df['sepallength']), np.poly1d(np.polyfit(iris_df['sepallength'], iris_df['petallength'], 1))
          (np.unique(iris_df['sepallength'])),color='red')
plt.title('correlation')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.show()

#
corr=iris_df.corr()
sns.heatmap(corr, cmap='viridis',annot=True, )
plt.show()

plt.scatter(iris_df['sepallength'], iris_df['sepalwidth'],color='grey')
plt.plot(np.unique(iris_df['sepallength']), np.poly1d(np.polyfit(iris_df['sepallength'], iris_df['sepalwidth'], 1))
          (np.unique(iris_df['sepallength'])),color='red')
plt.title('correlation')
plt.xlabel('sepal length')
plt.ylabel('petal width')
plt.show()


# #graph
# #histogram
# plt.hist(iris_df['sepallength'])
# plt.show()
#
# plt.hist(iris_df['sepallength'])
# plt.hist(iris_df['petallength'])
# plt.hist(iris_df['sepalwidth'])
# plt.hist(iris_df['petalwidth'])
# plt.show()
#
# # Pair Plot
# sns.pairplot(iris_df)
# plt.show()
#
# #sactter Plot
# plt.scatter(iris_df['sepalwidth'], iris_df['petalwidth'], iris_df['sepallength'], iris_df['petallength'])
# plt.show()
#
#
# #Box Plot
# newdf = iris_df[['sepalwidth','petalwidth','sepallength','petallength']]
#
# newdf.boxplot()
# plt.show()
#
# plt.boxplot(iris_df['sepalwidth'])
# plt.show()
# plt.boxplot(iris_df['petalwidth'])
# plt.show()
# plt.boxplot(iris_df['sepallength'])
# plt.show()
# plt.boxplot(iris_df['sepallength'])
# plt.show()
#
# #Parallel Chart
# parallel_coordinates(iris_df,'class', colormap = 'cool')
# plt.show()

#Distribution chart
# sns.displot(iris_df , hue = iris_df['class'], kind="kde", multiple="stack")
# plt.show()


# sns.violinplot(data=iris_df, x='class', y='sepallength')
# plt.show()
# sns.violinplot(data=iris_df, x='class', y='petalwidth')
# plt.show()
# sns.violinplot(data=iris_df, x='class', y='sepalwidth')
# plt.show()
# sns.violinplot(data=iris_df, x='class', y='petallength')
# plt.show()

sns.scatterplot(data=iris_df, x="class", y="sepallength", size = 20)
# show the graph
plt.show()




