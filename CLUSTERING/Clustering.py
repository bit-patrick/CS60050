# Roll 17EC30027
# Name Pratik Bhagwat
# Assignment Number 4
# K-Means Clustering

import numpy as np
import pandas as pd

data_set = pd.read_csv('data4_19.csv',names=['sl','sw','pl','pw','iris'])
X= data_set.iloc[:,:-1]

def Eucld(x,y):
    sum=0;
    for i in range(len(x)):
        sum+=(x[i]-y[i])**2
    sum=np.sqrt(sum)
    return sum
        
k = 3

print("Calculating Cluster Means: ")

C=[]
i=np.random.randint(0,len(X)//3)
C.append(list(X.iloc[i]))
j=np.random.randint(len(X)//3,2*len(X)//3)
C.append(list(X.iloc[j]))
h=np.random.randint(2*len(X)//3,len(X))
C.append(list(X.iloc[h]))   
C_old = [[0,0,0,0]]*len(C)   

clusters = np.zeros(len(X))
error=0;
for i in range(len(C)):
    error+=Eucld(C[i],C_old[i])

for rounds in range(10):
    for i in range(len(X)):
        distances=[]
        for j in range(len(C)):
            distances.append(Eucld(C[j],X.iloc[i]))
        mi=distances[0]
        for j in range(len(distances)):
            mi=min(distances[j],mi)
        for j in range(len(distances)):
            if(distances[j]==mi):
                clusters[i]=j
    for j in range(len(C)):
        C_old[j]=C[j]
    for i in range(k):
        points=[X.iloc[j] for j in range(len(X)) if clusters[j]==i]
        C[i]=list(np.mean(points,axis=0))
    error=0;
    for i in range(len(C)):
        error+=Eucld(C[i],C_old[i])
    if(rounds%3):
        print(".")
        
for i in range(k):
    for j in range(4):
        C[i][j]=float("{0:.2f}".format(C[i][j],2))
print("Centroids Calculated!!\n\n")
print("Calculating Jaccard Distances:")
col=list(np.unique(data_set['iris']))
print(".") 
numc=[0,0,0]
numm=[0,0,0]
for i in range(3):
    for j in range(len(X)):
        if data_set['iris'][j]==col[i]:
            numc[i]+=1
        if clusters[j]==i:
            numm[i]+=1


print(".")
Jaccard=[]
Cluster=['','','']  
for i in range(3):
    print(".")
    jac=0
    for j in range(3):
        inter=0
        for l in range(len(data_set)):
            if data_set['iris'][l]==col[j] and clusters[l]==i:
                inter+=1
        if jac<inter/(numc[j]+numm[i]-inter):
           jac = inter/(numc[j]+numm[i]-inter)
           Cluster[i]=col[j]
    Jaccard.append(abs(1-jac))
    
print("Jaccard Distances Calculated!!\n\n")    

for i in range(3):
    print("Mean Centroid for Cluster",i+1,":",C[i],"and it corresponds to",Cluster[i])
print("\n")
for i in range(3):
    print("Jaccard Distance for Cluster",i+1,"(",Cluster[i],"):",float("{0:.2f}".format(Jaccard[i])))
    