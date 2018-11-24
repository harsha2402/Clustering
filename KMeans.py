import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
from sklearn.decomposition import PCA

def get_distance_between_points(p1, p2):
    return np.sqrt(sum((p1 - p2) ** 2))


def get_closest_centroid_to_point(p, centroids):
    min_distance = sys.maxsize
    nearest_centroid = 0
    ind = 0
    for centroid in centroids:
        distance = np.sqrt(sum((p - centroid) ** 2))
        if distance < min_distance:
            min_distance = distance
            nearest_centroid = ind
        ind += 1
    return nearest_centroid


def calculate_new_centroid(list_of_values_cluster):
    r = len(list_of_values_cluster)
    c = len(list_of_values_cluster[0])
    updated_centroid = list()
    for index in range(0, c):
        sum_values = 0
        for j in range(0, r):
            sum_values += list_of_values_cluster[j][index]
        updated_centroid.append(sum_values/r)
    return updated_centroid


def k_means_clustering(dataframe, number_of_clusters):
    num_cols = dataframe.shape[1]
    num_rows = dataframe.shape[0]
    data = dataframe.iloc[:, 2:num_cols].values
    random_indices = np.random.randint(0, data.shape[0], number_of_clusters)
    initial_centroids = data[random_indices]
    old_centroids = list()
    num_of_iterations = 17
    curr_iteration = 0
    point_dictionary = dict()
    index_dictionary = dict()
    cluster_list = list()
    converged = False
    point_to_cluster_mapping = dict()

    for i in range(0, number_of_clusters):
        cluster_list.append(list())

    for i in range(0, number_of_clusters):
        cluster_list[i].append(initial_centroids[i])

    while True:
        if curr_iteration == num_of_iterations or converged:
            break
        curr_iteration += 1
        for i in range(0, num_rows):
            point = data[i]
            closest_centroid = get_closest_centroid_to_point(point, initial_centroids)
            frozen_array = frozenset(point)
            if (i + 1) in point_to_cluster_mapping.keys():
                point_to_cluster_mapping.pop(i + 1, None);
            if frozen_array in point_dictionary.keys() and index_dictionary.keys():
                cluster_index = point_dictionary[frozen_array]
                list_index = index_dictionary[frozen_array]
                if list_index < len(cluster_list[cluster_index]):
                    cluster_list[cluster_index].pop(list_index)
            for k in range(0, number_of_clusters):
                if k == closest_centroid:
                    point_dictionary[frozenset(point)] = k
                    point_to_cluster_mapping[i + 1] = k + 1
                    if len(cluster_list[k]) == 0:
                        index_dictionary[frozenset(point)] = 0
                    else:
                        index_dictionary[frozenset(point)] = len(cluster_list) - 1
                    cluster_list[k].append(point)
                    initial_centroids[k] = calculate_new_centroid(cluster_list[k])

        for i in range(0, number_of_clusters):
            if len(old_centroids) != 0 and get_distance_between_points(initial_centroids[i], old_centroids[i]) < 1:
                converged = True
        old_centroids = initial_centroids
    print(len(cluster_list[0]))
    print(len(cluster_list[1]))
    print(len(cluster_list[2]))
    print(len(cluster_list[3]))
    print(len(cluster_list[4]))
    return point_to_cluster_mapping



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

    print("Rand index is " + str(agree/total))

def main():
    dataframe = pd.read_csv('new_dataset_1.txt', sep='\t', header=None)
    num_cols = dataframe.shape[1]
    num_rows = dataframe.shape[0]
    df_whole = dataframe.iloc[:, 1:]
    ground_truth_mat = df_whole.as_matrix()
    groundTruth = ground_truth_mat[:, 0]
    ground_truth_dictionary = dict()
    truth_data = dataframe.iloc[:, 0:2].values
    for i in range(0, num_rows):
        ground_truth_dictionary[truth_data[i][0]] = truth_data[i][1]
    data = dataframe.iloc[:, 2:num_cols].values
    data_tsne = PCA(n_components=2).fit_transform(data)
    print(data_tsne[:, 0])

    colors_list  = list()
    res = list()
    point_dictionary = k_means_clustering(dataframe, 10)
    for key in point_dictionary.keys():
        res.append(point_dictionary[key])

    '''
    for i in range(0, num_rows):
        if point_dictionary[i + 1] == 1:
            colors_list.append('r')
        elif point_dictionary[i + 1] == 2:
            colors_list.append('g')
        elif point_dictionary[i + 1] == 3:
            colors_list.append('b')
        elif point_dictionary[i + 1] == 4:
            colors_list.append('y')
        elif point_dictionary[i + 1] == 5:
            colors_list.append('k')
    '''

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=res, alpha=0.8)
    print(rand_index(groundTruth, res))
    plt.show()



if __name__ == "__main__":
        main()


