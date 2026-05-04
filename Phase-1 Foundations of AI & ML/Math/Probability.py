
Copy

"""
PROBABILITY — Bayes theorem, distributions
==========================================
What it is:
Probability measures how likely something is to happen, on a scale
from 0 (impossible) to 1 (certain). Distributions describe how
probabilities are spread across all possible outcomes. Bayes theorem
is a formula for updating your belief about something when you get
new evidence.
 
Why it matters in AI:
Every model output is fundamentally a probability. When a model says
"this email is spam" it means P(spam | this email) = 0.92. Bayes
theorem is the mathematical foundation for how models update beliefs
given data. Distributions are used to initialise weights, model
uncertainty, and understand data. Probability is the language that
connects raw model outputs to real-world decisions.
 
Pipeline position:
  model output (logits) -> softmax -> probabilities -> [this file: interpret]
"""

import numpy as np

# Excerise 1 - Basic probability 
# P(event) = favourable outcomes / total outcomes

print("=" * 50)
print("1. BASIC PROBABILITY")
print("=" * 50)
 
# simulate 10000 coin flips
flips = np.random.choice(["heads", "tails"], size=10000)
p_heads = np.sum(flips == "heads") / len(flips)
 
print(f"10000 coin flips -> P(heads) = {p_heads:.4f}  (expect ~0.5)")
print()
 
# model confidence as probability
model_output = 0.87
print(f"model says P(spam) = {model_output}")
print(f"model says P(not spam) = {round(1 - model_output, 2)}")
print("probabilities always sum to 1.0")
print()

# Excerise 2 - Bayes theorem
# P(A|B) = P(B|A) * P(A) / P(B)
#
# In plain English:
# posterior = likelihood * prior / evidence
#
# Example: spam filter
# P(spam | "free money") = ?

print("=" * 50)
print("2. BAYES THEOREM")
print("=" * 50)
 
p_spam           = 0.30   # prior: 30% of all emails are spam
p_words_spam     = 0.80   # if spam, 80% chance it has "free money"
p_words_not_spam = 0.05   # if not spam, 5% chance it has "free money"
 
# total probability of seeing "free money" in any email
p_words = (p_words_spam * p_spam) + (p_words_not_spam * (1 - p_spam))
 
# bayes theorem -- update belief given evidence
p_spam_given_words = (p_words_spam * p_spam) / p_words
 
print("prior P(spam) =", p_spam)
print("P('free money' | spam) =", p_words_spam)
print("P('free money' | not spam) =", p_words_not_spam)
print()
print("after seeing 'free money':")
print("posterior P(spam | 'free money') =", round(p_spam_given_words, 4))
print()
print("belief updated from 30% to", round(p_spam_given_words * 100, 1), "%")
print("this is how naive bayes classifiers work")
print()

#Excerise 3 - Normal (Gaussian) distribution
# The ball curve. Most values cluster around the mean.
# Used to: initialise neural network weights, model
# netural variation in data, understand feature spread.

print("=" * 50)
print("3. NORMAL DISTRIBUTION")
print("=" * 50)
 
# generate 1000 samples from a normal distribution
samples = np.random.normal(loc=0.0, scale=1.0, size=1000)
#                           ^mean      ^std
 
print("1000 samples from N(mean=0, std=1):")
print(f"  actual mean: {round(np.mean(samples), 4)}  (expect ~0.0)")
print(f"  actual std:  {round(np.std(samples), 4)}   (expect ~1.0)")
print()
 
# neural networks initialise weights from a normal distribution
# so no single weight starts too large or too small
weights_init = np.random.normal(loc=0.0, scale=0.01, size=(3, 4))
print("randomly initialised weight matrix (3x4):")
print(weights_init.round(6))
print("small values near 0 -- prevents exploding activations at start of training")
print()

# Exercise 4 - Uniform distribution
# Every value in a range is equally likely.
# Also used for weight initialisation (Xavier/He init).

print("=" * 50)
print("4. UNIFORM DISTRIBUTION")
print("=" * 50)
 
samples_uniform = np.random.uniform(low=0.0, high=1.0, size=1000)
print("1000 samples from Uniform(0, 1):")
print(f"  min:  {round(np.min(samples_uniform), 4)}")
print(f"  max:  {round(np.max(samples_uniform), 4)}")
print(f"  mean: {round(np.mean(samples_uniform), 4)}  (expect ~0.5)")
print()
print("softmax outputs are probabilities -- they follow a distribution too")
print("understanding distributions helps you read model confidence correctly")