import pandas as pd
import numpy as np
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_distance_between_points(point1, point2):
    return np.sqrt(sum((point1 - point2) ** 2))

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

def main():
    dataframe = pd.read_csv('new_dataset_2.txt', sep='\t', header=None)

    df_whole = dataframe.iloc[:, 1:]
    ground_truth_mat = df_whole.as_matrix()
    groundTruth = ground_truth_mat[:, 0]

    num_cols = dataframe.shape[1]
    num_rows = dataframe.shape[0]
    data = dataframe.iloc[:, 2:num_cols].values
    initial_clusters = list()
    no_of_clusters =3
    for i in range(0, num_rows):
        cluster = list()
        cluster.append(i)
        initial_clusters.append(cluster)
    while len(initial_clusters) > no_of_clusters:
        print(len(initial_clusters))
        for i in range(0, len(initial_clusters)):
            min_distance = sys.maxsize
            target_cluster = -1
            if i >= len(initial_clusters):
                continue
            cluster_i = initial_clusters[i]
            for j in range(0, len(cluster_i)):
                point1 = cluster_i[j]
                for k in range(0, len(initial_clusters)):
                    if i != k:
                        cluster_k = initial_clusters[k]
                        for l in range(0, len(cluster_k)):
                            point2 = cluster_k[l]
                            distance = get_distance_between_points(data[point1], data[point2])
                            if distance < min_distance:
                                target_cluster = k
                                min_distance = distance
            for index in range(0, len(initial_clusters[target_cluster])):
                cluster_i.append(initial_clusters[target_cluster][index])
            initial_clusters.pop(target_cluster)
    print(initial_clusters)

    new_cluster = initial_clusters
    res=[]
    index = 0
    val = 1
    for nested_list in new_cluster:
        for num in range(len(nested_list)):
            res.append(val)
            index = index+1
        val = val+1

    print(res)
    rand_index(groundTruth, res)

    data_tsne = PCA(n_components=2).fit_transform(data)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=res, alpha=0.8)
    plt.show()


if __name__ == "__main__":
    main()
