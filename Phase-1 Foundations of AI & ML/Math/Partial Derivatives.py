"""
PARTIAL DERIVATIVES AND GRADIENTS
==================================
What it is:
A partial derivative measures how much the output changes when you
move one input variable slightly, while holding all others fixed.
A gradient is the collection of all partial derivatives for a
function with multiple inputs -- one slope per input variable.
 
Why it matters in AI:
A neural network has millions of weights. Each weight is one input
variable to the loss function. The gradient of the loss is a vector
containing one partial derivative per weight. Each value answers:
if I change this one weight slightly, how much does the loss change?
Gradient descent then moves every weight simultaneously using this
information. Without partial derivatives you could not train a network
with more than one weight.
 
Pipeline position:
  loss -> [this file: partial derivatives -> gradient] -> update all weights
"""
 
import numpy as np

# Exercise 1 - Partial derivative by hand
# f(x, y) = x**2 + 3+y
# df/dx = 2x (derivative w.r.t x, treat y as constant)
# df/dy = 3 (derivative w.r.t. y, treat x as constant)

def f(x, y):
    return x**2 + 3*y
 
def partial_x(x, y):
    tiny = 0.0001
    return (f(x + tiny, y) - f(x, y)) / tiny  # y held fixed
 
def partial_y(x, y):
    tiny = 0.0001
    return (f(x, y + tiny) - f(x, y)) / tiny  # x held fixed
 
x, y = 3.0, 2.0
 
print("=" * 50)
print("1. PARTIAL DERIVATIVES")
print("=" * 50)
print(f"f(x,y) = x^2 + 3y    at x={x}, y={y}")
print(f"f({x},{y}) = {f(x,y)}")
print()
print(f"df/dx = {round(partial_x(x, y), 4)}  (expect 6.0 = 2*3)")
print(f"df/dy = {round(partial_y(x, y), 4)}  (expect 3.0 always)")
print()
print("df/dx tells us: increase x by 1 -> loss increases by ~6")
print("df/dy tells us: increase y by 1 -> loss increases by ~3")
print()

# Excerise 2 - The gradient vector
# Collect all partial derivatives into one vector.
# gradient = [df/dx, df/dy]
# Points in the direction of steepest increase.
# Gradient descent moves in the OPPOSITE direction.

gradient = np.array([partial_x(x, y), partial_y(x, y)])
 
print("=" * 50)
print("2. GRADIENT VECTOR")
print("=" * 50)
print(f"gradient at (x={x}, y={y}): {gradient.round(4)}")
print("this vector points in the direction of steepest increase")
print("gradient descent moves opposite to this -- downhill")
print()

# Excerise 3 - Gradient descent with two weights 
# Same as colculus.py but now we update two weights simultaneously.
# Each weight gets its own partial derivative

print("=" * 50)
print("3. GRADIENT DESCENT WITH TWO WEIGHTS")
print("=" * 50)
 
x, y = 4.0, 5.0
learning_rate = 0.1
 
for step in range(15):
    grad_x = partial_x(x, y)
    grad_y = partial_y(x, y)
    x = x - learning_rate * grad_x  # update x using its own gradient
    y = y - learning_rate * grad_y  # update y using its own gradient
    loss = f(x, y)
    print(f"step={step+1:02d}  x={x:.4f}  y={y:.4f}  loss={loss:.4f}")
 
print()
print("x converges to 0 (minimum of x^2)")
print("y keeps decreasing because df/dy=3 always -- no minimum, just a slope")
print()

# Excerise 4 - Gardient in a neural network context
# A simple one-neuron network.
# loss = (prediction - true)^2
# prediction = weights * input
# dl/dw = partial derivative of loss w.r.t, weight 

print("=" * 50)
print("4. GRADIENT IN A NEURON")
print("=" * 50)
 
x_input = 2.0    # input feature
y_true  = 6.0    # correct answer
w       = 1.0    # starting weight
 
learning_rate = 0.05
 
for step in range(20):
    prediction = w * x_input
    loss       = (prediction - y_true) ** 2
 
    # partial derivative of loss w.r.t. weight
    # dL/dw = 2 * (prediction - true) * x_input
    grad_w = 2 * (prediction - y_true) * x_input
 
    w = w - learning_rate * grad_w
    print(f"step={step+1:02d}  w={w:.4f}  pred={round(w*x_input,4):.4f}  loss={loss:.4f}")
 
print()
print(f"final weight: {round(w, 4)}  (expect ~3.0 because 3*2=6=y_true)")