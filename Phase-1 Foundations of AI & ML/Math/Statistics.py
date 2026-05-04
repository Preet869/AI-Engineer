"""
STATISTICS — mean, variance, standard deviation
================================================
What it is:
Statistics gives you tools to summarise and understand your data
before you ever touch a model. Mean tells you the centre. Variance
tells you how spread out the data is. Standard deviation is variance
in the original units so you can actually read it.
 
Why it matters in AI:
You cannot train a good model on data you do not understand.
These three numbers tell you if your data is balanced, if features
are on similar scales, and if something looks wrong before training.
Standard deviation is also used in normalisation (standardisation)
which is the most common preprocessing step in real ML pipelines.
 
Pipeline position:
  raw data -> [this file: understand your data] -> normalise -> train
"""

import numpy as np

# Exercise 1 - Mean
# The average value. 
# In ML: tells you the centre of each feature.

rating = np.array([3.2, 4.5, 2.8, 4.9, 3.6, 4.1, 2.5, 4.8])

mean = np.mean(rating)
print("=" * 50)
print("1. Mean")
print("=" * 50)
print("rating", rating)
print("mean", round(mean, 4))
print("Meaning: The average movie rating is this dataset")
print()

# Exercise 2 - Variance 
# How spread out the values are from the mean.
# Formula: average of squared differences from the mean.
# In ML: high veriance means your data is scattered widly.

variance = np.var(rating)
print("=" * 50)
print("2. VARIANCE")
print("=" * 50)
print("variance", round(variance, 4))
print("meaning: How spread out rating are from the mean")
print("note: units are squared (ratings sqaured) -- hard to interpret directly")
print()

# Exercise 3 - Standard Deviation
# Sqaure root of variance. Same units as original data.
# In ML: the most useful spread measure -- readable and
# used directly in standardisation (z-score normalisation).

std = np.std(rating)
print("=" * 50)
print("3. STANDARD DEVIATION")
print("=" * 50)
print("std:", round(std, 4))
print("meaning: most rating fall within", round(mean - std, 2), "and", round(mean + std, 2))
print()

# Excerise 4 - Standardistaion (z-score normalisation)
# this is the real-world use of mean and std in ML.
# Instead of dividing by max (min-max normalisation),
# you subtract mean and divide by std.
# Result: data centred at 0, spread of 1.
# this is what standardScaler does in scikit-learn.

rating_standardised = (rating - mean) / std
print("=" * 50)
print("4. STANDARDISATION (z-score)")
print("=" * 50)
print("original:        ", rating)
print("standardiesed:    ", rating_standardised.round(4))
print("new mean:", round(np.mean(rating_standardised), 4)) # should be -0
print("new std:  ", round(np.std(rating_standardised), 4)) # shoudl be -1
print()
print("data is now centred at 0 with a spread of 1")
print("no feature dominates due to scale -- same goal as min-max but smarter")

# Excerise 5 - Apply to a full dataset matrix 
# In real ML you standarise every column (feature) seperately. 
movies = np.array([
    [3.2, 120, 5000],
    [4.5, 150, 950000],
    [2.8, 130, 12000],
    [4.9, 180, 870000],
    [3.6, 140, 34000],
])

col_means = movies.mean(axis=0) # Mean of each column
col_stds = movies.std(axis=0) # std od each column

movies_standardised = (movies - col_means) / col_stds
print()
print("=" * 50)
print("5. STANDARDISE A FULL DATASET")
print("=" * 50)
print("originals:")
print(movies)
print("Standardised:")
print(movies_standardised.round(4))
print()
print("each column now has mean=0 and std=1")
print("this is exactly what sklearn's StandardScaler does")