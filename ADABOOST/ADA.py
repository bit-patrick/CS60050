# Roll 17EC10003
# Name Anand Raj
# Assignment Number 3
# Adaptive Boost

import numpy as np
import pandas as pd
from numpy import log2 as log


epis = np.finfo(float).eps

data_set = pd.read_csv('data3_19.csv')
#test_set = pd.read_csv('data3_19.csv')
test_set = pd.read_csv('test3_19.csv',skiprows=0,names=['pclass','age','gender','survived'])
#Train_data_set = data_set.sample(frac=0.75, random_state=99)  
#Test_data = data_set.loc[~data_set.index.isin(Train_data_set.index), :]
#Test_data = Test_data.reset_index(drop=True)
#Train_data_set = Train_data_set.reset_index(drop=True)
Data_Table=data_set
#test_set.iloc[0]=['pclass','age','gender','survived']
Test_Table=test_set


#Changing Yes/No to 1/0.
for i in range(len(Data_Table)):
    if Data_Table['survived'][i]=='yes':
        Data_Table['survived'][i]=1
    else:
        Data_Table['survived'][i]=-1
for i in range(len(Test_Table)):
    if Test_Table['survived'][i]=='yes':
        Test_Table['survived'][i]=1
    else:
        Test_Table['survived'][i]=-1

#Add initial weights to the Data Set    
w=[1/len(Data_Table)]*len(Data_Table)
Data_Table['weight']=w

"""Preprocessing"""                                             
"""##############################################################################################################"""
"""Decision Tree Modified With Weights"""
#table=Train_data_set #(As required)#
#This function returns the entropy of the current table

def Entropy_Table(table):
    Class=table.keys()[-2]
    entropy1=0
    values=table[Class].unique()
    for value in values:
        fraction = 0
        for i in range(len(table)):
            if table[Class][i]==value:
                fraction+=table['weight'][i]
        entropy1+=-fraction*np.log2(fraction)
    return abs(entropy1)

#This function returns the entropy of the given attribute
def Entropy_Table_Attribute(table,attribute):
    Class=table.keys()[-2]
    target_variables=table[Class].unique()
    variables=table[attribute].unique()
    entropy2=0
    for variable in variables:
       entropy3=0
       for target_variable in target_variables:
           num=0
           den=0
           for i in range(len(table)):
               num+=(table[attribute][i]==variable)*(table[Class][i]==target_variable)*table['weight'][i]
               den+=(table[attribute][i]==variable)*table['weight'][i]
           fraction=num/(den+epis)
#           print(variable," ",target_variable," ",num," ",den)
           entropy3 += -fraction*log(fraction+epis)
       fraction2=den
       entropy2 += -fraction2*entropy3
    return abs(entropy2)

#This function returns the best information gain attribte 
def Winner_Attribute(table):
    IG=[]
    for key in table.keys()[:-2]:
        IG.append(Entropy_Table(table)-Entropy_Table_Attribute(table,key))
    return table.keys()[:-2][np.argmax(IG)]

#This function returns the subtable
def Sub_Table(table,node,value):
    return (table[table[node]==value].reset_index(drop=True)).drop(node,axis=1)

#This function draws a tree as a dictionary
def Create_Tree(table,tree=None):
    at=table.columns
    
    if len(at)==2:
        return tree
    #Class=table.keys()[-1]
    node=Winner_Attribute(table)
    
    att_values=np.unique(table[node])
    table1=table
    #print(table1)
    #print('Yo',node)
    if tree is None:
        tree={}
        tree[node]={}
        
    for value in att_values:
        #print(value)
        subtable=Sub_Table(table1,node,value)
        clValue=np.unique(subtable['survived'])
        if len(clValue)==1:
            counts=[0]
            for i in range(len(subtable)):
                counts[0]+=subtable['weight'][i]
        elif len(clValue)==2:
            counts=[0,0]
            for i in range(len(subtable)):
                if subtable['survived'][i]==clValue[0]:
                    counts[0]+=subtable['weight'][i]
                else:
                    counts[1]+=subtable['weight'][i]
        counts=counts/sum(counts)
        if len(counts)==1:
            tree[node][value]=clValue[0]
        elif len(at)==3 and counts[0]/counts[1]>1:#(According To Taste)
            tree[node][value]=clValue[0]
            
        elif len(at)==3 and counts[1]/counts[0]>1:#(According To Taste) 
            tree[node][value]=clValue[1]
        else:
            tree[node][value]=Create_Tree(subtable)
    
    return tree

"""Decision Tree End"""
"""##############################################################################################################"""
"""Prediction And Error"""

#This function creates the prediction array
def Predict(data,tree):
    keys=tree.keys()
    for i in range(len(keys)):
        key=keys[i]
        value=data[key]
        #print(type(key),type(value),type(tree))
        if value not in tree[key]:
            return -1
        #Missing values
        
        tree=tree[key][value]
        #prediction = '0'
        if type(tree) is dict:
            prediction = Predict(data,tree)
            
        else:
            prediction= tree
            break;
    
    return prediction

#Prediction array
def Predict_Array(data_set,Tree):
    
    pred = []
    for i in range(len(data_set)):
        data=data_set.iloc[i]
        pred.append(Predict(data,Tree))
    return pred

#Error Function    
def Get_Error_Rate(pred, Y):
    count=0
    #return sum(pred!=Y)/float(len(Y))    
    for i in range(len(pred)):
        if pred[i]!=Y[i]:
            count+=1
    return count/len(pred)

"""Prediction and Error End"""
"""##############################################################################################################"""    
"""AdaBoost"""
    
#AdaBoost Function        
def AdaptiveBoost(X_train,Y_train,X_test,Y_test,m):
    
    n_train, n_test= len(X_train), len(X_test)
    wtr=np.ones(n_train)/n_train
    pred_test= np.zeros(n_test)
   
    
    for i in range(m):
        Tree=Create_Tree(Data_Table)
        pred_train_i = Predict_Array(X_train,Tree)
        pred_test_i = Predict_Array(X_test,Tree)
#        if pred_train[0]==Y_train[0]:
#            print("Happy")
#        else:
#            print("Sad")
        miss = [int(x) for x in (pred_train_i != Y_train)]
        miss2 = [x if x==1 else -1 for x in miss]
        
        err_m = np.dot(wtr,miss)/sum(wtr)
        alpha_m = 0.5*np.log((1-err_m+epis)/float(err_m+epis))
        
        wtr=np.multiply(wtr,np.exp([float(x) * alpha_m for x in miss2]))
        wtr=wtr/sum(wtr)
        Data_Table['weight']=wtr
        pred_test = [sum(x) for x in zip(pred_test,[float(x) * (1-err_m) for x in pred_test_i])]

    pred_test= np.sign(pred_test)
    
    return Get_Error_Rate(pred_test, Y_test)

"""AdaBoost End"""
"""##############################################################################################################"""
"""Driver Function"""

#PreProcessing
Tree = Create_Tree(Data_Table)
X_train, Y_train = Data_Table.iloc[:,:-2],Data_Table.iloc[:,-2]
X_test, Y_test = Test_Table.iloc[:,:-1],Test_Table.iloc[:,-1]

print("\nRunning Adaboost for upto 5 iterations\n")
for i in range(5):
    Er_i = AdaptiveBoost(X_train,Y_train,X_test,Y_test,i+1) 
    print("Accuracy on Test Data (Adaboost for ",i+1," iterations) = ",(1-Er_i)*100,"%")
    

