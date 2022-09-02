
from scipy import spatial
import math
import numpy

book_A = numpy.array([2, 3])
book_B = numpy.array([4, 1])

# ---- Easy Solution ----
cosine_dist_easy = spatial.distance.cosine(book_A, book_B)

# ---- Manuel Solution ----
# Dot product between vectors
product_AB = numpy.dot(book_A, book_B)
# Length of vectors
book_A_length = numpy.linalg.norm(book_A,2)
book_B_length = numpy.linalg.norm(book_B,2)
# Similarity measures by cosine distance
cosine_dist_manuel = 1 - (product_AB / (book_A_length * book_B_length))

print(cosine_dist_easy)
print(cosine_dist_manuel)