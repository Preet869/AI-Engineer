"""
Core vector concepts for ML with working examples.

Concepts covered:
  1. What a vector is
  2. Vector magnitude (length)
  3. Normalisation - why and how (preprocessing before training)
  4. Dot product - similarity between vectors
"""

import numpy as np


# 1. WHAT A VECTOR IS
# A vector is an ordered list of numbers.
# Each number is one feature of your data.
# The model only sees these numbers, not labels or context.

movie = np.array([8.5, 1990, 148, 950000])
#                  ^     ^     ^      ^
#               rating  year  mins  reviews

print("=" * 50)
print("1. RAW VECTOR")
print("=" * 50)
print("movie =", movie)
print("shape =", movie.shape)
print("dtype =", movie.dtype)
print()


# 2. VECTOR MAGNITUDE (LENGTH)
# Magnitude = the size of the vector.
# Formula: sqrt(x1^2 + x2^2 + x3^2 + ...)
# This is Pythagoras extended to N dimensions.

magnitude = np.linalg.norm(movie)

print("=" * 50)
print("2. MAGNITUDE")
print("=" * 50)
print("||movie|| =", round(magnitude, 2))
print("Note: dominated by 950000 (reviews)")
print("-- that is the scaling problem normalisation fixes.")
print()


# 3. THE PROBLEM WITH RAW DATA
# Each feature lives on a completely different scale:
#   rating   ->  0 to 10
#   year     ->  1900 to 2024
#   runtime  ->  0 to 300
#   reviews  ->  0 to 1,000,000
#
# Dot products treat bigger numbers as more significant.
# 950,000 dominates the model learning -- not because
# reviews matter most, but because the number is largest.

print("=" * 50)
print("3. THE SCALING PROBLEM (visual)")
print("=" * 50)
features = ["rating  ", "year    ", "runtime ", "reviews "]
for name, val in zip(features, movie):
    bar = "=" * int(val / 50000)
    print(" ", name, str(round(val)).rjust(10), " ", bar if bar else "(too small to show)")
print()
print("reviews completely dominates.")
print("Model would learn reviews matter most by accident.")
print()


# 4. NORMALISATION - FIX THE SCALE
# Divide each feature by its maximum possible value.
# Every feature now lives in the range 0 to 1.
# Same information, no accidental scale dominance.
# This is what you do BEFORE feeding data into a model.

max_values = np.array([10, 2024, 300, 1_000_000])
movie_normalised = movie / max_values

print("=" * 50)
print("4. NORMALISED VECTOR")
print("=" * 50)
print("original   =", movie)
print("normalised =", movie_normalised)
print()
print("Feature by feature:")
for name, raw, norm in zip(features, movie, movie_normalised):
    print(" ", name, str(round(raw)).rjust(10), " ->", round(norm, 4))
print()
print("Now all features compete on equal footing.")
print()


# 5. DOT PRODUCT - VECTOR SIMILARITY
# Multiply matching elements, sum the results.
# High score = vectors point in a similar direction = similar meaning.
#
# Used in ML for:
#   - attention  (query . key = how much to attend to a token)
#   - recommendations  (user . movie = match score)
#   - every neuron in a neural network
#
# IMPORTANT: only makes sense on normalised vectors.
# On raw data the biggest-scale feature dominates every score.

user_preferences = np.array([0.9, 0.5, 0.6, 0.3])
#                              ^     ^     ^     ^
#                           rating  year  mins  reviews

movie_a = np.array([0.85, 0.98, 0.49, 0.95])
movie_b = np.array([0.40, 0.60, 0.90, 0.10])

score_a = np.dot(user_preferences, movie_a)
score_b = np.dot(user_preferences, movie_b)

print("=" * 50)
print("5. DOT PRODUCT - RECOMMENDATION")
print("=" * 50)
print("user preferences =", user_preferences)
print("movie A          =", movie_a)
print("movie B          =", movie_b)
print()
print("score_a =", round(score_a, 4))
print("score_b =", round(score_b, 4))
print()
winner = "Movie A" if score_a > score_b else "Movie B"
print("Recommend:", winner, " (higher dot product = better match)")

print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print("vector      -> ordered list of numbers = one data sample")
print("magnitude   -> length of the vector (Pythagoras in N dims)")
print("normalise   -> scale all features to 0-1 BEFORE training")
print("dot product -> multiply + sum two vectors = similarity score")
print()
print("Pipeline:  raw data -> [normalise] -> train model -> evaluate")