
"""
17EC30027
Pratik Pyarelal Bhagwat
Assignment Number 2
Naive Bayes Classifier
"""

import pandas as pd
 

dataset = pd.read_csv('data2_19.csv')
test = pd.read_csv('test2_19.csv')

#data preprocessing
dataset = dataset["D,X1,X2,X3,X4,X5,X6"].str.split(",", n = 6, expand = True) 
test = test["D,X1,X2,X3,X4,X5,X6"].str.split(",", n = 6, expand = True)
   
dataset=dataset.astype(int)
test=test.astype(int)

dataset.columns=['D','X1','X2','X3','X4','X5','X6']
test.columns=['D','X1','X2','X3','X4','X5','X6']


test_X = test.iloc[:,1:]
test_y = test.iloc[:,0]


def Yes_No(dataset,Val):
    '''
    function calculattes probability of Total Yes or NO
    '''
    target = dataset.loc[:,"D"]
    sum = 0
    for i in range(len(target)):
        if(target[i]==Val):
                sum = sum+1
    return sum/len(target)

Yes = Yes_No(dataset,1)
No = Yes_No(dataset,0)

def cal_prob(dataset,attribute,X,value):
    '''
    (dataset,attribute,X,value);
    dataset is dataset;
    attribute is col name as in X1,X2....;
    X is your 1 2 3 4 5 ;
    value if person is happy(1) or not(0);
    '''
    Col  = dataset.loc[:,attribute]
    target = dataset.loc[:,"D"]
    total = 0
    count = 0
    for i in range(len(Col)):
        if(target[i]==value):
            total = total+1
            if(Col[i] == X):
                count = count+1
    return (count+1)/(total+5)        #laplasian smoothing adding 1 above and classes(5) below 


#Calculations store the values of all probabilities 
Calculations = {0:{},1:{}}
labels = [1,2,3,4,5]
for i in dataset.columns[1:]:        
    Calculations[0][i] = {}
    Calculations[1][i] = {}
    for category in labels:
        Calculations[0][i][category] = cal_prob(dataset,i,category,0)
        Calculations[1][i][category] = cal_prob(dataset,i,category,1)


#PRediction array for our test data
Prediction=[]

for row in range(0,len(test_X)):
    prod_0 = No
    prod_1 = Yes
    for feature in test_X.columns:
        prod_0 *= Calculations[0][feature][test_X[feature].iloc[row]]
        prod_1 *= Calculations[1][feature][test_X[feature].iloc[row]]
        
        #Predict the outcome
    if prod_0 > prod_1:
        Prediction.append(0)
    else:
        Prediction.append(1)
        
tp,tn = 0,0
for j in range(0,len(Prediction)):
    if Prediction[j] == 0:
        if test_y.iloc[j] == 0:
            tp += 1
    else:
        if test_y.iloc[j] == 1:
            tn += 1
            
print ("accuracy on test data = ",(tp+tn)*100/14," %")


