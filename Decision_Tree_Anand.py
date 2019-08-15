# Roll 17EC10003
# Name Anand Raj
# Assignment Number 1
#



import numpy as np
import pandas as pd
from numpy import log2 as log
import pprint
epis = np.finfo(float).eps

data_set = pd.read_csv('data1_19.csv')

#If you want to break the data in Train and Test#
#75% of the data to train and rest to test in a random manner#
#Train_data_set = data_set.sample(frac=0.75, random_state=99)  
#Test_data = data_set.loc[~data_set.index.isin(Train_data_set.index), :]
#Test_data = Test_data.reset_index(drop=True)
#Train_data_set = Train_data_set.reset_index(drop=True)
table=data_set #Or 
#table=Train_data_set #(As required)#


#Entropy of the current table.    
def func1(table):
    Class=table.keys()[-1]
    entropy=0
    values=table[Class].unique()
    for value in values:
        fraction=table[Class].value_counts()[value]/len(table[Class])
        entropy+=-fraction*np.log2(fraction)
    return entropy


#Entropy of a particular attribute.
def func2(table,attribute):
    Class=table.keys()[-1]
    target_variables=table[Class].unique()
    variables=table[attribute].unique()
    entropy2=0
    for variable in variables:
       entropy=0
       for target_variable in target_variables:
           num = len(table[attribute][table[attribute]==variable][table[Class] ==target_variable])
           den = len(table[attribute][table[attribute]==variable])
           fraction=num/(den+epis)
           entropy += -fraction*log(fraction+epis)
       fraction2=den/len(table)
       entropy2 += -fraction2*entropy
    return abs(entropy2)


#Return best information gain attribte 
def func3(table):
    IG=[]
    for key in table.keys()[:-1]:
        IG.append(func1(table)-func2(table,key))
    return table.keys()[:-1][np.argmax(IG)]

#Return Subtable
def func4(table,node,value):
    return (table[table[node]==value].reset_index(drop=True)).drop(node,axis=1)

#Draw a tree as a dictionary
def func5(table,tree=None):
    at=table.columns
    #print(at)
    if len(at)==1:
        return tree
    #Class=table.keys()[-1]
    node=func3(table)
    #print(node)
    att_values=np.unique(table[node])
    table1=table
    #print(table1)
    #print('Yo',node)
    if tree is None:
        tree={}
        tree[node]={}
        
    for value in att_values:
        #print(value)
        subtable=func4(table1,node,value)
        clValue,counts=np.unique(subtable['survived'],return_counts=True)
        #print(clValue)
        #print(counts)
        if len(counts)==1:
            tree[node][value]=clValue[0]
        elif counts[0]/counts[1]>100 or (len(at)==2 and counts[0]/counts[1]>1):#(According To Taste)
            tree[node][value]=clValue[0]
            
        elif counts[1]/counts[0]>100 or (len(at)==2 and counts[1]/counts[0]>1):#(According To Taste) 
            tree[node][value]=clValue[1]
        else:
            tree[node][value]=func5(subtable)
    
    return tree

tree=func5(table)

#Print the tree
pprint.pprint(tree)


#Create the prediction array
def func6(dat,tree):
    for nodes in tree.keys():
        
        value=dat[nodes]
        
        if value not in tree[nodes]:
            return 'No'
        #Missing values
        
        tree=tree[nodes][value]
        prediction = 0
        
        if type(tree) is dict:
            prediction = func6(dat,tree)
            
        else:
            prediction= tree
            break;
    
    return prediction

#Prediction array
pred = []

for i in range(len(data_set)):
    data=data_set.iloc[i]
    pred.append(func6(data,tree))
    

count=0
#Check score
for i in range(len(pred)):
    if pred[i]==data_set.survived[i]:
        count+=1
        
#print(count/len(pred))
        
     
    
            

            

