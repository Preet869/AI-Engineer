"""
MATRIX OPERATIONS — multiplication, transpose, inverse
=======================================================
What it is:
Beyond basic matrix multiplication (covered in matrix.py), three
operations come up constantly in ML: multiply (combining data with
weights), transpose (flipping rows and columns), and inverse
(the matrix equivalent of division -- undoing a transformation).
 
Why it matters in AI:
Transpose is used in backpropagation -- when the gradient flows
backwards through a layer it passes through the transposed weight
matrix. Inverse is used in mathematical derivations and some
classical ML algorithms like linear regression solved analytically.
Understanding these gives you the full picture of how data flows
forwards and backwards through a network.
 
Pipeline position:
  forward pass uses matrix multiply
  backward pass uses transpose of the same weight matrices
"""

import numpy as np

# Excerise 1 - Matrix multiplication (recap + shape rule)
# (m x n) @ (n x p) = (m x p)
# inner diemsions must match

print("=" * 50)
print("1. MATRIX MULTIPLICATION")
print("=" * 50)

X = np.array([          # 4 samples x 3 features
    [0.8, 0.5, 0.9],
    [0.3, 0.7, 0.4],
    [0.6, 0.2, 0.8],
    [0.9, 0.4, 0.6],
])
 
W = np.array([          # 3 features -> 2 neurons
    [0.5, 0.1],
    [0.3, 0.8],
    [0.2, 0.6],
])
 
output = X @ W          # (4x3) @ (3x2) = (4x2)

print("X shape:", X.shape, " W shape:", W.shape, " output shape:", output.shape)
print("output (one row per sample, one col per neuron):")
print(output.round(4))
print()


# Excerise 2 - Transpose
# Flip rows and columns. (m x n) become (n x m)
# Notation: W.T or W^T
# In ML: used in back propagation -- gradient flows
# backwards through W.T, or W.

print("=" * 50)
print("2. TRANSPOSE")
print("=" * 50)

W = np.array([
    [0.5, 0.1],
    [0.3, 0.8],
    [0.2, 0.6],
])
 
print("W shape:", W.shape)
print("W:")
print(W)
print()
print("W.T shape:", W.T.shape)
print("W.T:")
print(W.T)
print()
print("rows became columns, columns became rows")
print("in backprop: gradient = error @ W.T  (flows backwards through transposed weights)")
print()

# Exercise 3 - Inverse 
# The matrix equivalent of dividing by a number.
# A @ A_inv = identity matrix (ones on diagonal, zeros elsewhere)
# only works on sqaure matrices that are not singular.
# In ML: used in analytical solution for linear regression.

print("=" * 50)
print("3. INVERSE")
print("=" * 50)
 
A = np.array([
    [2.0, 1.0],
    [5.0, 3.0],
])
 
A_inv = np.linalg.inv(A)
 
print("A:")
print(A)
print()
print("A inverse:")
print(A_inv.round(4))
print()
print("A @ A_inv (should be identity matrix):")
print((A @ A_inv).round(4))
print()

# Excerise 4 - Full forward and backward pass shapes
# Show how transpose is used in back propagation.

print("=" * 50)
print("4. FORWARD AND BACKWARD PASS SHAPES")
print("=" * 50)

# forward pass
X  = np.random.rand(4, 3)   # 4 samples, 3 features
W  = np.random.rand(3, 2)   # 3 features -> 2 neurons
Z  = X @ W                  # (4x3) @ (3x2) = (4x2)
 
# pretend this is the gradient of the loss w.r.t. Z
dZ = np.random.rand(4, 2)   # same shape as Z
 
# backward pass -- how gradient flows back to W and X
dW = X.T @ dZ               # (3x4) @ (4x2) = (3x2) -- same shape as W
dX = dZ @ W.T               # (4x2) @ (2x3) = (4x3) -- same shape as X
 
print("forward pass:  X", X.shape, "@ W", W.shape, "= Z", Z.shape)
print("backward pass: X.T", X.T.shape, "@ dZ", dZ.shape, "= dW", dW.shape)
print("backward pass: dZ", dZ.shape, "@ W.T", W.T.shape, "= dX", dX.shape)
print()
print("dW is how we update W -- same shape as W, one gradient per weight")
print("this is what PyTorch computes automatically with loss.backward()")