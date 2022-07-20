from __future__ import absolute_import
import sys
sys.path.insert(0, ""); sys.path.insert(0, "../")
import numpy
import math
from scipy import spatial
from SharedFunctionality.Visualization import Visualizer


def similarity_euclidean(vec_a, vec_b):
    vector_substraction = numpy.array(vec_a) - numpy.array(vec_b)
    return numpy.linalg.norm(vector_substraction)

def similarity_cosine(vec_a, vec_b):
    return spatial.distance.cosine(vec_a, vec_b)

def calc_similarities(vectors, dist_func):
    similarities = []
    for index_a, vector_a in enumerate(vectors):
        for vector_b in vectors[index_a+1:]:
            similarity = dist_func(vector_a, vector_b)
            similarities += [{"similarity": similarity, "vector_a": vector_a, "vector_b": vector_b}]
    return similarities



# ---------- Define Measurements ----------
# Vector = [question x, question y]
vec_a = [1, 3]
vec_b = [2, 2]
vec_c = [2, 3]
vec_d = [4, 4]
vec_e = [5, 5]
vec_f = [4, 6]
all_vectors = [vec_a, vec_b, vec_c, vec_d, vec_e, vec_f]
print(f"Vectors: {all_vectors}\n")



# --------------------- Euclidean Similarity ---------------------

# ----------- Find Euclidean Similarity -----------
# Distance between x coordinates
a_b_x_dist = abs(vec_a[0] - vec_b[0])
# Distance between y coordinates
a_b_y_dist = abs(vec_a[1] - vec_b[1])
# Use pythagoras to find distance between points
a_b_euclidean_dist = math.sqrt(a_b_x_dist**2 + a_b_y_dist**2)
# Similarity is measured by euclidean distance
sim_a_b_euc = a_b_euclidean_dist
print(f"Euclidean similarity A,B: {sim_a_b_euc}")


# ---------- Dynamic Euclidean Similarity ---------
# Distance in all dimensions
b_c_distances = [abs(vec_b[i]-vec_c[i]) for i in range(0, len(vec_b))]
# Sum together all distances squared (pythagoras)
sum_of_squared_distances = sum([dist**2 for dist in b_c_distances])
# Find distance
sim_b_c_euc = math.sqrt(sum_of_squared_distances)
print(f"Euclidean similarity B,C: {sim_b_c_euc}")


# -------- Simplified Euclidean Similarity --------
sim_c_d_euc = numpy.linalg.norm(numpy.array(vec_c) - numpy.array(vec_d))
print(f"Euclidean similarity C,D: {sim_c_d_euc}")


# ---------- Put into a method for reuse ----------
sim_d_e_euc = similarity_euclidean(vec_d, vec_e)
print(f"Euclidean similarity D,E: {sim_d_e_euc}")


# ------------ Find highest similarity ------------
similarities = []
# Go through all vector pairs
for index_a, vector_a in enumerate(all_vectors):
    # sim(a, b) == sim(b, a)
    # Start from the next vector to not calculate the same pair twice
    for index_b, vector_b in enumerate(all_vectors[index_a+1:]):
        similarity = similarity_euclidean(vector_a, vector_b)
        similarities += [{"similarity": similarity, "vector_a": vector_a, "vector_b": vector_b}]

# --- Smaller distance means higher similarity ---
highest_euc_sim = min(similarities, key=lambda x: x["similarity"])
print(f"Highest euclidean Similarity: {highest_euc_sim}")


# ------------ Find lowest similarity -------------
# Higher distance means lower similarity
lowest_euc_sim = max(similarities, key=lambda x:x["similarity"])
print(f"Lowest euclidean Similarity: {lowest_euc_sim}\n")





# ----------------------- Cosine Similarity ----------------------

# ------------ Find Cosine Similarity -------------
# Length of vectors
vec_a_length = math.sqrt(vec_a[0]**2 + vec_a[1]**2)
vec_b_length = math.sqrt(vec_b[0]**2 + vec_b[1]**2)
# Dot product of vectors
vector_a_b_product = vec_a[0]*vec_b[0] + vec_a[1]*vec_b[1]
# Similarity measures by cosine distance
sim_a_b_cos = vector_a_b_product / (vec_a_length * vec_b_length)
print(f"Cosine similarity A,B: {sim_a_b_cos}")


# ---------- Dynamic Euclidean Similarity ---------
# Length of vectors
b_length = math.sqrt(sum(value**2 for value in vec_b))
c_length = math.sqrt(sum(value**2 for value in vec_c))
# Dot product between vectors
b_c_product = sum([vec_b[i] * vec_c[i] for i in range(0, len(vec_b))])
# Similarity measures by cosine distance
sim_b_c_cos = b_c_product / (b_length * c_length)
print(f"Cosine similarity B,C: {sim_b_c_cos}")


# ---------- Simplified Cosine Similarity ---------
sim_c_d_cos = spatial.distance.cosine(vec_c, vec_d)
print(f"Cosine similarity C,D: {sim_c_d_cos}")


# ---------- Put into a method for reuse ----------
sim_d_e_cos = similarity_cosine(vec_d, vec_e)
print(f"Cosine similarity D,E: {sim_d_e_cos}")


# ------------ Find highest similarity ------------
similarities = calc_similarities(all_vectors, similarity_cosine)


# --- Smaller distance means higher similarity ---
highest_cos_sim = min(similarities, key=lambda x: x["similarity"])
print(f"Highest euclidean Similarity: {highest_cos_sim}")


# ------------ Find lowest similarity -------------
# Higher distance means lower similarity
lowest_cos_sim = max(similarities, key=lambda x:x["similarity"])
print(f"Lowest euclidean Similarity: {lowest_cos_sim}\n")




# ----------- Visualize points -----------
Visualizer.vectors_2D(all_vectors)