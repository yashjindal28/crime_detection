import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

df1 = pd.read_csv('dataset_part1.csv')
##so I HAVE ALL the dataset with the following crimes to be removed.

##cleaning the dataset1 for including the crimes
##the dataset_split1 which we chose from df2 will be treated

def cleandata():
    ctr = 0
    ctr1 = 0
    for ele in df1.iloc[:,2]:
        f = 0
        print("The crime in dataset :: ",ele)
        for elem in crimes:
            if elem == ele:
                    print(elem,"  " ,ele)
                    f=1
                    break
        print("THe flag val :: " , f)
        if f == 0: 
            print("Dropping initiated")
            df1.drop(df1.index[ctr],inplace=True)
        if f == 1:
            print("NO drop needed")
            ctr += 1
        print(ctr," ",f)
        ctr1+=1
    print("Greg")
    print("Count of ones :: ",ctr1)


cleandata()


df1.to_csv('data1.csv')


##generalising the dataset
##the reduced dataset is being generalised in this attempt

# this will replace "Boston Celtics" with "Omega Warrior" 
#df.replace(to_replace ="Boston Celtics", 
                 #value ="Omega Warrior") 

def generalizeData():
    ctr = 0
    ctr1 = 0
    for elem in df1.iloc[:,2]:
        print(elem)
        for ele,val in crime_dictionary.items():
            print(ele,"  " , val)
            df1.replace(to_replace=elem, 
                        value = val,inplace=True)

generalizeData()


pd.set_option('display.max_columns',None)

X_df = pd.read_csv('./finaalllly.csv')

X_df1 = X_df.iloc[:,1:]

X_df2 = X_df.iloc[:,1:]

y_df = pd.read_excel('./CRIME_CODE.xlsx')
y_df_cpy = y_df
x_df_cpy = X_df1

y_df['ID'].value_counts(normalize=True)

X_df2['Location'].dtype

obj_cols = [c for c in X_df2.columns if X_df2[c].dtype == 'O']

for c in obj_cols:
    X_df2[c] = pd.factorize(X_df2[c])[0]
    
feats = [c for c in X_df2.columns if c != 'OFFENSE_CODE']

X = X_df2[feats]
y = X_df2['OFFENSE_CODE']

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

preds = logreg.predict(X_test)

accuracy_score(y_test, preds)
##We got the accuracy of 78% .



