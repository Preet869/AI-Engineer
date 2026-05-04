"""
LOSS FUNCTIONS — log loss, cross-entropy, softmax
==================================================
What it is:
A loss function measures how wrong your model is. It takes the
model's prediction and the true answer and returns one number.
The smaller that number, the better the model. Training is the
process of making this number as small as possible.
 
Why it matters in AI:
The loss function is what the model is actually trying to minimise.
It defines what "correct" means. Cross-entropy is the standard loss
for classification problems. Softmax converts raw model outputs into
probabilities so cross-entropy can compare them to true labels.
Log loss and cross-entropy are the same thing -- different names
used in different contexts.
 
Pipeline position:
  prediction -> [this file: measure wrongness] -> derivatives -> update weights
"""

import numpy as np

# Excerise 1 - Why log
# Log shrinks large numbers and stretches small ones.
# A confident wrong prediction gets punished heavily.
# a confident correct prediction gets rewarded mildly.
# This asymetry is exactly what you want in training.

print("=" * 50)
print("1. WHY LOG?")
print("=" * 50)

probs = [0.01, 0.1, 0.5, 0.9, 0.99]
for p in probs:
    print(f"  prob={p:.2f} log(p)={round(np.log(p), 4)}")

print()
print("small probability (wrong and confident) -> very negative log -> huge penalty")
print("large probability (right and confident) -> near zero log    -> small penalty")
print()

# Excerise 2 - Log loss (binary classification)
# Used when model outputs one probability (yes/no problems).
# Formula: -( y*log(p) + (1-y)*log(1-p) )
#   y = true label (0 or 1)
#   p = predicted probability of being 1

def log_loss(y_true, y_pred):
    # clip to avoid log(0) which is undefined
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print("=" * 50)
print("2. LOG LOSS (binary)")
print("=" * 50)

cases = [
    (1, 0.95, "correct and confident"),
    (1, 0.55, "correct but uncertain"),
    (1, 0.10, "wrong and confident"),
    (0, 0.05, "correct and confident"),
    (0, 0.90, "wrong and confident"),
]

for y, p, label in cases:
    loss = log_loss(y, p)
    print(f"true={y}  pred={p:.2f}  loss={round(loss, 4):>8}  ({label})")

print()

# Excerise 3 - Softmax
# Converts raw model outputs (logits) into probabilities.
# All outputs become postitive and sum to exactly 1.0.
# Formula: exp(x) / sum(exp(x))

def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps)

print("=" * 50)
print("3. SOFTMAX")
print("=" * 50)

# raw outputs from a model (logits) -- 3 classess: cat, dog, bird
logits = np.array([2.1, 0.5, -0.3])

probs = softmax(logits)

print("logits (raw model output):", logits)
print("probabilities after softmax:", probs.round(4))
print("sum of probabilities:", round(probs.sum(), 4))  # always 1.0
print()
print("class predictions:")
classes = ["cat", "dog", "bird"]
for cls, p in zip(classes, probs):
    print(f"  {cls}: {round(p * 100, 1)}%")
print()


# Excerise 4 - Cross-entropy loss (multi-class)
# used after softmax for classification with 3+ classes.
# Formual: -sum( y_true * log(y_pred))
# y_true is a one-hot vector: [0, 1, 0] means class 1 is correct.

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1.0)
    return -np.sum(y_true * np.log(y_pred))
 
print("=" * 50)
print("4. CROSS-ENTROPY LOSS")
print("=" * 50)
 
# true label is dog (index 1) -- one-hot encoded
y_true = np.array([0, 1, 0])
 
# good prediction -- model is confident about dog
good_pred = softmax(np.array([0.2, 3.0, 0.1]))
# bad prediction -- model thinks it is a cat
bad_pred  = softmax(np.array([3.0, 0.2, 0.1]))
 
print("true label: dog  (one-hot:", y_true, ")")
print()
print("good prediction:", good_pred.round(4), "-> loss:", round(cross_entropy(y_true, good_pred), 4))
print("bad  prediction:", bad_pred.round(4),  "-> loss:", round(cross_entropy(y_true, bad_pred), 4))
print()
print("higher loss = more wrong = bigger gradient signal = bigger weight update")