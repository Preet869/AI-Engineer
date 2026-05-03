import numpy as np

# ─────────────────────────────────────────────
# Exercise 1 - Build a dataset matrix
# ─────────────────────────────────────────────
# Each row = one movie
# Each column = one feature: [rating, runtime, reviews]

"""
MATRICES
========
A matrix is a grid of numbers — rows and columns. In ML your entire
dataset is a matrix. Each row is one sample. Each column is one feature.
A neural network weight layer is also a matrix.

Why it matters:
Instead of processing one sample at a time, a matrix lets you process
your entire dataset in one operation. Matrix multiplication is the core
computation of a neural network forward pass — every layer transforms
its input by multiplying it through a weight matrix. Understanding matrix
shapes and multiplication is the difference between debugging PyTorch
errors in seconds versus hours.

What this file covers:
- building a dataset matrix with NumPy
- accessing rows (samples) and columns (features)
- normalising an entire dataset in one line
- matrix multiplication with @ — running many dot products at once
- the shape rule — why inner dimensions must match, the most common
  error in deep learning

Pipeline position:
  raw data -> normalise matrix -> [this file: matrix multiply through layers] -> output
"""

movies = np.array([
    [2.4, 120, 500],
    [3.2, 150, 600],
    [2.8, 130, 550],
    [3.6, 140, 650],
    [2.9, 125, 520],
])

print("=" * 50)
print("1. DATASET MATRIX")
print("=" * 50)
print(movies)
print("shape:", movies.shape)  # (5, 3) = 5 movies, 3 features
print()


# ─────────────────────────────────────────────
# Exercise 2 - Access rows and columns
# ─────────────────────────────────────────────

print("=" * 50)
print("2. ROWS AND COLUMNS")
print("=" * 50)

# row 0 = first movie, all its features
print("first movie (row 0):", movies[0])

# row -1 = last movie
print("last movie  (row -1):", movies[-1])

# [:, 0] = all rows, column 0 = every movie's rating
# in ML you do this to inspect or scale one feature at a time
print("all ratings (col 0):", movies[:, 0])
print()


# ─────────────────────────────────────────────
# Exercise 3 - Normalise the whole dataset
# ─────────────────────────────────────────────
# Divide the entire matrix by max values in one line.
# NumPy applies the division column by column automatically.
# This is what a real preprocessing pipeline does.

max_values = np.array([3.6, 150, 650])
movies_normalised = movies / max_values

print("=" * 50)
print("3. NORMALISED MATRIX")
print("=" * 50)
print("original:")
print(movies)
print("normalised:")
print(movies_normalised.round(4))
print()


# ─────────────────────────────────────────────
# Exercise 4 - Matrix multiplication
# ─────────────────────────────────────────────
# weights = how much a user cares about each feature
# (5 x 3) @ (3,) = (5,)  -- one score per movie
# each score = dot product of that movie with the weights
# this is exactly what a single neural network neuron does

weights = np.array([0.5, 0.3, 0.2])

scores = movies_normalised @ weights

print("=" * 50)
print("4. PREDICTED INTEREST SCORES")
print("=" * 50)
print("weights (rating, runtime, reviews):", weights)
print("scores per movie:", scores.round(4))
print("best match: movie", scores.argmax(), "(highest score)")
print()


# ─────────────────────────────────────────────
# Exercise 5 - Spot the shape rule
# ─────────────────────────────────────────────

print("=" * 50)
print("5. SHAPE RULE")
print("=" * 50)

# WRONG: (5 x 3) @ (5 x 1) -- inner dims are 3 vs 5, no match
weights_wrong = np.array([[0.5], [0.3], [0.2], [0.1], [0.9]])

try:
    result = movies_normalised @ weights_wrong
except ValueError as e:
    print("Error (expected):", e)
    print()

# RIGHT: (5 x 3) @ (3 x 1) -- inner dims are both 3
# shape rule: (5 x 3) @ (3 x 1) = (5 x 1)
#                   ^   ^
#                   must match
weights_right = np.array([[0.5], [0.3], [0.2]])

result = movies_normalised @ weights_right
print("result shape:", result.shape)   # (5, 1) - one score per movie
print("scores:", result.round(4))