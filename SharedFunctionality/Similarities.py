import numpy
import scipy

class Similarity:
    def Euclidean(vector_a, vector_b):
        return numpy.linalg.norm(vector_a - vector_b)

    def Cosine(vector_a, vector_b):
        return scipy.spatial.distance.cosine(vector_a, vector_b)