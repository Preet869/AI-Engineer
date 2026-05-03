import numpy as np

"""
DOT PRODUCTS
============
A dot product takes two vectors, multiplies their matching elements,
and sums the result into a single number. That number measures how
similar the two vectors are — how much they point in the same direction.

Why it matters:
The dot product is the fundamental operation inside all of ML. Every
neuron in a neural network computes a dot product. Attention in a
transformer is a dot product between query and key vectors. A
recommendation score is a dot product between a user vector and an
item vector. Matrix multiplication is just many dot products running
simultaneously. Once you understand the dot product, you understand
what a model is actually doing at every step.

Key rule — only meaningful after normalisation:
A dot product on raw data measures scale, not similarity. A movie with
900,000 reviews scores higher than a perfect match simply because the
number is larger. Normalise first, then compute dot products.

What this file covers:
- computing a dot product manually step by step
- using np.dot() to verify
- dot product as a similarity score between vectors
- matrix multiplication as many dot products at once
- the angle interpretation — positive means similar, zero means no
  relation, negative means opposite
- why raw dot products are misleading without normalisation

Pipeline position:
  normalised data -> [this file: dot products inside every layer] -> prediction
"""

# Excerise 1: One dot product by hand

a = [0.8, 0.5, 0.3]
b = [0.6, 0.4, 0.9]

result = 0
for i in range(len(a)):
    result += a[i] * b[i]

print(result)
print("np.dot result:", np.dot(a, b))

# Exercise 2 - Dot product as similarity
# each vector = [action_score, romance_score, scifi_score]
action  = [0.9, 0.2, 0.8]
romance = [0.2, 0.9, 0.1]
scifi   = [0.8, 0.1, 0.9]

user = [0.8, 0.1, 0.7]

# compute similarity between user and each movie
score_action  = np.dot(user, action)
score_romance = np.dot(user, romance)
score_scifi   = np.dot(user, scifi)

print("action  score:", round(score_action, 4))
print("romance score:", round(score_romance, 4))
print("scifi   score:", round(score_scifi, 4))

# find the best match
scores = {"action": score_action, "romance": score_romance, "scifi": score_scifi}
best = max(scores, key=scores.get)
print("recommend:", best)

# a high dot product means the two vectors point in a similar direction
# meaning the user's preferences and the movie's features align closely

# Excerise 3 - Maxtrix of dot products
movies = np.array([
    [2.4, 120, 500],
    [3.2, 150, 600],
    [2.8, 130, 550],
    [3.6, 140, 650],
    [2.9, 125, 520],
])

weights = np.array([0.5, 0.3, 0.2])
# add this before the @ multiplication
max_values = movies.max(axis=0)
movies_normalised = movies / max_values
scores = movies_normalised @ weights
print("scores", scores)

# Excerise 4 - Dot product and angle
# pair 1 — pointing in similar direction
a1 = np.array([0.9, 0.8, 0.7])
b1 = np.array([0.8, 0.9, 0.6])
print("similar vectors:", round(np.dot(a1, b1), 4))

# pair 2 — pointing in opposite directions
a2 = np.array([1.0, 0.0, 0.0])
b2 = np.array([-1.0, 0.0, 0.0])
print("opposite vectors:", round(np.dot(a2, b2), 4))

# a dot product of 0 means the vectors are perpendicular --
# they share nothing in common, zero similarity.
# positive = similar direction, negative = opposite direction.

# Excerise 5 - Without normalisation

movie_a = np.array([4.9, 180, 90000])
movie_b = np.array([5.0, 190, 100000])
user    = np.array([0.9, 0.5, 0.3])

# RAW dot products -- before normalisation
score_a_raw = np.dot(user, movie_a)
score_b_raw = np.dot(user, movie_b)

print("RAW scores:")
print("  movie_a:", round(score_a_raw, 2))
print("  movie_b:", round(score_b_raw, 2))
print()

# NORMALISED dot products -- after scaling to 0-1
max_values   = np.array([5.0, 240, 1_000_000])
movie_a_norm = movie_a / max_values
movie_b_norm = movie_b / max_values

score_a_norm = np.dot(user, movie_a_norm)
score_b_norm = np.dot(user, movie_b_norm)

print("NORMALISED scores:")
print("  movie_a:", round(score_a_norm, 4))
print("  movie_b:", round(score_b_norm, 4))
print()

print("movie_a_norm:", movie_a_norm.round(4))
print("movie_b_norm:", movie_b_norm.round(4))

# the raw dot product is misleading because 90000 and 100000
# dominate the calculation completely -- rating and runtime
# barely contribute. the model would rank movies by review
# count, not by actual match to the user's preferences.
# normalisation puts all features on equal footing first.