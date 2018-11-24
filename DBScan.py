import numpy as np
from scipy import spatial
import pandas as pd
import math
import collections
from sklearn.metrics import jaccard_similarity_score
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# return all points within P's eps-neighborhood (including P)
def regionQuery(data, P, eps):

    withinEpsNeighborhood = []

    #Iterate through all the datapoints
    for datapoint in range(0, len(data)):
        #Calculate euclidean distance
        #dist = np.linalg.norm(data[datapoint]- data[P])
        dist = np.sqrt(np.sum(data[datapoint]- data[P])**2)
        if dist <= eps:
            withinEpsNeighborhood.append(datapoint)


    return withinEpsNeighborhood


def expandCluster(data, clusterLabel, visited, p, neighborPts, c, eps, minpts):
    #add p to cluster first
    clusterLabel[p] = c
    for pPrime in neighborPts:
        if pPrime not in visited:
            visited.append(pPrime)
            neighborPtsPrime = regionQuery(data, pPrime, eps)
            if len(neighborPtsPrime) >= minpts:
                # To do: NeighborPts = NeighborPts joined with NeighborPts'
                neighborPts.extend(neighborPtsPrime)
            if pPrime not in clusterLabel:
                clusterLabel[pPrime] = c

def DBScan(data, eps, minpts):
    visited = []
    clusterLabel = np.zeros(len(data)) #Will store the final cluster label assignment

    #start with cluster id 0
    c = 0

    for p in range(0, len(data)):
        if p not in visited:
            visited.append(p)
            neighborPts = regionQuery(data, p, eps)
            if len(neighborPts) < minpts:
                #mark p as noise
                clusterLabel[p] = -1
            else:
                c = c+1
                expandCluster(data, clusterLabel, visited, p, neighborPts, c, eps, minpts)

    return clusterLabel

'''
def jaccardCoefficient(x, y):
    
    return jaccard_similarity_score(x, y) #CAlling the library function for Jaccard. Used only for testing

#My implementation of Jaccard Coefficient. This is being called later
def compute_jaccard_index(set1, set2):
    count = 0
    for i in range(len(set1)):
        if set1[i] == set2[i]:
            count = count + 1
    return count/len(set1)
'''

def rand_index(groundTruth, res):
    w, h = len(res), len(res);

    mat_groundTruth = [[0 for x in range(w)] for y in range(h)]
    mat_res = [[0 for x in range(w)] for y in range(h)]

    for i in range(0,len(groundTruth)):
        for j in range(0,len(groundTruth)):
            if i==j:
                mat_groundTruth[i][j] = 1
            elif groundTruth[i] == groundTruth[j]:
                mat_groundTruth[i][j] = 1
                mat_groundTruth[j][i] = 1
            else:
                mat_groundTruth[i][j] = 0
                mat_groundTruth[j][i] = 0

    for i in range(0,len(res)):
        for j in range(0,len(res)):
            if i==j:
                mat_res[i][j] = 1
            elif res[i] == res[j]:
                mat_res[i][j] = 1
                mat_res[j][i] = 1
            else:
                mat_res[i][j] = 0
                mat_res[j][i] = 0


    #print(np.asmatrix(res))

    agree = 0
    total = 0
    for i in range(0,len(res)):
        for j in range(0,len(res)):

            total = total+1
            if mat_groundTruth[i][j] == mat_res[i][j]:
                agree = agree+1

    print(agree/total)


df_a = pd.read_csv('new_dataset_1.txt',sep='\t', header=None)
df1 = df_a.iloc[:,2:]
df_whole = df_a.iloc[:,1:]

ground_truth_mat = df_whole.as_matrix()
dataMat = df1.as_matrix()

groundTruth = ground_truth_mat[:,0]
print(len(groundTruth))

res = DBScan(dataMat,1.5,4) #Calling my implementation of DBScan




data_tsne =PCA(n_components=2).fit_transform(dataMat)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=res, alpha=0.8)
plt.show()

rand_index(groundTruth, res)