
"""
17EC30027
PRATIK BHAGWAT
ASSIGNMENT 1 DECISION TREE
"""

# Importing the libraries
import numpy as np
import pandas as pd
import math as m
import pprint

epis=np.finfo(float).eps #it is a small number to avoid divide by zero error

# Importing the dataset
dataset = pd.read_csv('data1_19.csv')
X_train = dataset.iloc[:,:3]
y_train = dataset.iloc[:,-1]
survi = dataset.iloc[:,-1]


#calculates the entropy of a column of a table
#cal_entropy(dataset,pclass) will calculate the entropy of pclass column
def cal_entropy(table,column):
    current=table.loc[:,column]
    values = list(current.unique())
    tentropy=0
    Y_N = list(table.iloc[:,-1].unique())
    for V in values:
        entropy=0
        for YN in Y_N:
            Numerator = len(table[column][table[column]==V][table.iloc[:,-1] ==YN])
            Denominator = len(table[column][table[column]==V])
            Fract=Numerator/(Denominator+epis)
            entropy += -Fract*np.log2(Fract+epis)
        Fract2=Denominator/len(table)
        tentropy += -Fract2*entropy
    
    return abs(tentropy)


#calculates the entropy of an entire table
def cal_entropy_init (table):
    pp = 0
    pn = 0
    pp=(table.iloc[:,-1]=="yes").sum()
    pn=(table.iloc[:,-1]=="no").sum()
    total = pp + pn
    pp = pp/(total+epis)
    pn = pn/(total+epis)
    entropy = -1*(pp*(np.log2(pp))+pn*(np.log2(pn)))
    return entropy

#gives us the best attribute to which we will split
#we use our infogain here i.e entropy of table - entropy after splitting of a column
def get_best(table):
    Infogain=[]
    for heads in table.keys()[:-1]:
        Infogain.append(cal_entropy_init(table)-cal_entropy(table,heads))
        
    best=np.argmax(Infogain)
    return table.keys()[:-1][best]

#after the data has been split on a particular attribute we need to make a new table again
#the find the best attribute of these table
def delete_table(table,node,value):
    return (table[table[node]==value].reset_index(drop=True)).drop(node,axis=1)

#these funtion builds a tree as a dictonary
def build_tree(table,tree=None):
   names=table.columns
   if len(names)==1:
        return tree
   node=get_best(table)
   att_values=np.unique(table[node])
   newtable=table
   if tree is None:
        tree={}
        tree[node]={}
        
   for value in att_values:
        #print(value)
       subtable=delete_table(newtable,node,value)
       clValue,counts=np.unique(subtable['survived'],return_counts=True)
       if len(counts)==1:
           tree[node][value]=clValue[0]
       elif counts[0]/counts[1]>100 or (len(names)==2 and counts[0]/counts[1]>1):
           tree[node][value]=clValue[0]
            
       elif counts[1]/counts[0]>100 or (len(names)==2 and counts[1]/counts[0]>1): 
            tree[node][value]=clValue[1]
       else:
           tree[node][value]=build_tree(subtable)
    
   return tree

#building the tree on the dataset
Tree = build_tree(dataset)
pprint.pprint(Tree, depth=6, width=50)

#Create the prediction array
def Predict(data,Tree):
    for splits in Tree.keys():
        value=data[splits]
        
        if value not in Tree[splits]:
            return 'No'
        
        Tree=Tree[splits][value]
        prediction = 0
        
        if type(Tree) is dict:
            prediction = Predict(data,Tree)
            
        else:
            prediction= Tree
            break;
    
    return prediction

#Prediction array
our_pred = []

for i in range(len(dataset)):
    data=dataset.iloc[i]
    our_pred.append(Predict(data,Tree))
    

count=0

#if our_pred matches the survived column
for i in range(len(our_pred)):
    if our_pred[i]==dataset.survived[i]:
        count+=1
        
accuracy = (count/len(our_pred))

    
    
    
    
    
    
    


    
    

    



    


    
    
