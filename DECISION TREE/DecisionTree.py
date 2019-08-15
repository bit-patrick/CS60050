
"""
17EC30027
PRATIK BHAGWAT
ASSIGNMENT 1 DECISION TREE
"""

# Importing the libraries
import numpy as np
import pandas as pd
import math as m

epis=np.finfo(float).eps

# Importing the dataset
dataset = pd.read_csv('data1_19.csv')
X_train = dataset.iloc[:,:3]
y_train = dataset.iloc[:,-1]
survi = dataset.iloc[:,-1]



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


def get_best(table):
    Infogain=[]
    for heads in table.keys()[:-1]:
        Infogain.append(cal_entropy_init(table)-cal_entropy(table,heads))
        
    best=np.argmax(Infogain)
    return table.keys()[:-1][best]


def delete_table(table,node,value):
    return (table[table[node]==value].reset_index(drop=True)).drop(node,axis=1)

def tree_depth(tree): 
    max_now = [] 
    string_tree = str(tree) 
    count_in = count_out =0
    for i in string_tree: 
        if count_out<count_in :
            if i == "{": 
                count_in += 1
            elif i == '}':
                count_out +=1
    
        else:
            max_now.append(count_in)
    return (max_now.max()) 

def Tree(table,tree = None):
    AT = X_train.columns
    if len(AT) == 1 :
        return tree
    node=get_best(table)
    value=np.unique(table[node])
    tablecpy=table
    
    if tree==None:
        tree = {}
        tree[node]={}
    
    for V in value:
        newtable = delete_table(tablecpy,node,V)
        clValue,counts=np.unique(newtable["survived"],return_counts=True)     
        if(len(counts)==1 or tree_depth == 3):
            tree[node][value]=clValue[0]
        else:     
            tree[node][value]=Tree(newtable)
            
    
    return tree

#def dict_depth(dic, level = 1): 
#       
#    str_dic = str(dic) 
#    counter = 0
#    for i in str_dic: 
#        if i == "{": 
#            counter += 1
#    return(counter) 
def build_tree(table,tree=None):
    at=table.columns
    #print(at)
    if len(at)==1:
        return tree
    #Class=table.keys()[-1]
    node=get_best(table)
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
        subtable=delete_table(table1,node,value)
        clValue,counts=np.unique(subtable['survived'],return_counts=True)
        print(clValue)
        print(counts)
        if len(counts)==1:
            tree[node][value]=clValue[0]
        elif counts[0]/counts[1]>250 or (len(at)==2 and counts[0]/counts[1]>1):
            tree[node][value]=clValue[0]
            
        elif counts[1]/counts[0]>250 or (len(at)==2 and counts[1]/counts[0]>1):
            tree[node][value]=clValue[1]
        else:
            tree[node][value]=build_tree(subtable)
    
    return tree
Tree = build_tree(dataset)
Tree = Tree(dataset)
str(tree)
tree = Tree(dataset)

    
    
    
    
    
    
    


    
    

    



    


    
    
