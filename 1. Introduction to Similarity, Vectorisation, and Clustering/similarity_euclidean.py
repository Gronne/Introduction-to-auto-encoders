import numpy
import math

book_A = numpy.array([2, 3])
book_B = numpy.array([4, 1])

book_A_B = book_A - book_B

# ---- Easy Solution ----
euclidean_dist_easy = numpy.linalg.norm(book_A_B)

# ---- Manuel Solution ----
# Sum together all distances squared (pythagoras)
sum_of_squared_distances = numpy.dot(book_A_B, book_A_B)
# Find distance
euclidean_dist_manuel = math.sqrt(sum_of_squared_distances)


print(euclidean_dist_easy)
print(euclidean_dist_manuel)
