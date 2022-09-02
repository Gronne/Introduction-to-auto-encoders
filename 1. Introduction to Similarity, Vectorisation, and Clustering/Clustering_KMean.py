from __future__ import absolute_import
import sys
sys.path.insert(0, ""); sys.path.insert(0, "../")
import numpy
from sklearn import cluster
from SharedFunctionality.Visualization import Visualizer
from SharedFunctionality.Similarities import Similarity


def vecs_dim_range(vectors, dimension):
    return (min([vector[dimension] for vector in vectors]), max([vector[dimension] for vector in vectors]))

def generate_random_clusters(ranges, n_points):
    random_vectors = numpy.random.rand(len(ranges), n_points)
    for index, row in enumerate(random_vectors):
        range_diff = ranges[index][1] - ranges[index][0]
        random_vectors[index] = row*range_diff + ranges[index][0]
    return random_vectors

def closest_cluster(clusters, vector, dist_func):
    distance_to_clusters = [dist_func(cluster, numpy.transpose(vector)) for cluster in clusters]
    return distance_to_clusters.index(min(distance_to_clusters))

def split_data_points_between_clusters(clusters, vectors, dist_func):
    clusters_vectors = [[] for _ in clusters]
    for vector in vectors:
        closest_cluster_index = closest_cluster(clusters, vector, dist_func)
        clusters_vectors[closest_cluster_index] += [vector]
    return clusters_vectors

def calc_mean_clusters(clusters_vectors):
    return [numpy.mean(cluster_vectors, axis=0) for cluster_vectors in clusters_vectors]




# -------------- Define Measurements --------------
# Vector = [question x, question y]
vec_a = [2, 3]
vec_b = [3, 1]
vec_c = [7, 5]
vec_d = [8, 7]
vec_e = [3, 8]
vec_f = [4, 9]
vec_g = [9, 2]
vec_h = [10, 1.8]
data_points = [vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g, vec_h]



# --------------- K-Mean clustering ---------------
# Define range for data_points
x_coor = [data_point[0] for data_point in data_points]
range_x = (min(x_coor), max(x_coor))
y_coor = [data_point[1] for data_point in data_points]
range_y = (min(y_coor), max(y_coor))
print(f"range X: {range_x}")
print(f"range Y: {range_y}")

# Create random start position for clusters
n_clusters = 2
clusters = generate_random_clusters([range_x, range_y], n_clusters)
print(f"Clusters: {clusters}")

#Calculate new start position in N iterations
iterations = 5
for _ in range(0, iterations):
    # Split data_points between clusters
    clusters_data_points = split_data_points_between_clusters(clusters, data_points, Similarity.Euclidean)
    # Calculate mean of each vector set as the the cluster positions
    clusters = calc_mean_clusters(clusters_data_points)

#Print final cluster positions
print(f"Clusters: {clusters}")



# ---------- K-Mean clustering simplified ----------
clusters = cluster.KMeans(n_clusters=n_clusters).fit(data_points).cluster_centers_
print(clusters)
