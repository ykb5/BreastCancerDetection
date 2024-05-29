import numpy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# reading data from the file
df=pd.read_csv("D:\VS Code\BreastCancerDetection\data.csv")

# printing dataset
print(df)
print('\n*\n*\n')

# return all the columns with null values count
print(df.isna().sum())

# remove the column
df=df.dropna(axis=1)
print('\n*\n*\n')

# Get the count of malignant<M> and Benign<B> cells
print(df['diagnosis'].value_counts())
print('\n*\n*\n')

# label encoding(convert the value of M and B into 1 and 0)
labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)

sns.pairplot(df.iloc[:,1:5],hue="diagnosis")