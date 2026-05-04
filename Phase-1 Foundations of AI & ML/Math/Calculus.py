"""
CALCULUS
========
Calculus is how a model learns. It answers one question at every
training step: for each weight, does increasing it make the model
more wrong or less wrong, and by exactly how much?

Why it matters:
Without calculus a model has no way to improve. It would have to
guess every weight randomly and hope. Derivatives give the model
a precise signal at every step — the slope tells it which direction
to move each weight and how far. This is the entire mechanism behind
training any neural network.

Key concepts:
- derivative: the slope of a function at one specific point.
  In ML, slope of the loss with respect to one weight.
- gradient descent: follow the slope downhill, step by step,
  until the loss stops getting smaller.
- chain rule: how the slope signal travels backwards through
  every layer of the network. This is backpropagation.
- learning rate: how big each step is. Too large and you
  overshoot. Too small and training takes forever.

What this file covers:
- approximating derivatives with finite differences
- gradient descent from scratch with a loop
- seeing how learning rate changes training behaviour
- chain rule computed by hand then verified
- PyTorch autograd doing all of the above automatically

Pipeline position:
  forward pass -> loss -> [this file: derivatives + chain rule] -> update weights -> repeat
"""

import torch
# Exercise 1 - Derivative by hand

# the original function
def f(x):
    return x ** 2

# finite difference — how computers approximate a derivative
# formula: (f(x + tiny) - f(x)) / tiny
# as tiny gets smaller, this gets closer to the true derivative
def derivative(x):
    tiny_number = 0.0001
    return (f(x + tiny_number) - f(x)) / tiny_number

# try at three different points
print("x=2,  derivative:", round(derivative(2), 4))   # expect ~4.0
print("x=0,  derivative:", round(derivative(0), 4))   # expect ~0.0  (flat — minimum)
print("x=-3, derivative:", round(derivative(-3), 4))  # expect ~-6.0

# the derivative at x=2 is 4.0 — slope is positive, move left to go downhill
# the derivative at x=0 is 0.0 — flat, this is the minimum, nowhere to go
# the derivative at x=-3 is -6.0 — slope is negative, move right to go downhill
# in ML each weight is an x, and gradient descent reads this slope
# to decide which direction to nudge the weight

# Excerise 2: Gradient descent from scratch
learning_rate = 0.1
x = 10.0
for step in range(20):
    grad = derivative(x)          # slope at current x
    x = x - learning_rate * grad  # take one step downhill
    loss = f(x)                   # how wrong are we now
    print(f"step={step+1:02d}  x={x:.4f}  loss={loss:.4f}")

# with learning_rate=0.9 the steps would be huge and x would
# overshoot the minimum, bouncing back and forth and possibly
# never settling — this is why learning rate matters so much

# Exercise 3 - Learning rate comparison

learning_rates = [0.01, 0.1, 0.9]

for lr in learning_rates:
    x = 10.0        # reset starting point for every run
    steps_to_converge = None

    print(f"\nlearning rate = {lr}")
    print("-" * 40)

    for step in range(20):
        grad = derivative(x)
        x = x - lr * grad
        loss = f(x)
        print(f"  step={step+1:02d}  x={x:.4f}  loss={loss:.6f}")

        # record the first step that gets below 0.01 loss
        if loss < 0.01 and steps_to_converge is None:
            steps_to_converge = step + 1

    if steps_to_converge:
        print(f"reached loss < 0.01 at step {steps_to_converge}")
    else:
        print("never reached loss < 0.01 in 20 steps")

# lr=0.01 is too small  -- takes many steps, may not converge in 20
# lr=0.1  is just right -- steady descent, converges cleanly
# lr=0.9  is too large  -- overshoots the minimum, bounces around
# in a real model you would pick 0.1 or tune it carefully

# Exercise 3 - Learning rate comparison

learning_rates = [0.01, 0.1, 0.9]

for lr in learning_rates:
    x = 10.0        # reset starting point for every run
    steps_to_converge = None

    print(f"\nlearning rate = {lr}")
    print("-" * 40)

    for step in range(20):
        grad = derivative(x)
        x = x - lr * grad
        loss = f(x)
        print(f"  step={step+1:02d}  x={x:.4f}  loss={loss:.6f}")

        # record the first step that gets below 0.01 loss
        if loss < 0.01 and steps_to_converge is None:
            steps_to_converge = step + 1

    if steps_to_converge:
        print(f"reached loss < 0.01 at step {steps_to_converge}")
    else:
        print("never reached loss < 0.01 in 20 steps")

# lr=0.01 is too small  -- takes many steps, may not converge in 20
# lr=0.1  is just right -- steady descent, converges cleanly
# lr=0.9  is too large  -- overshoots the minimum, bounces around
# in a real model you would pick 0.1 or tune it carefully

# Exercise 5 - PyTorch autograd
# ── part 1: simple derivative ──────────────────────
# x**2 has derivative 2x
# at x=3, derivative should be 6.0

x = torch.tensor(3.0, requires_grad=True)
# requires_grad=True tells PyTorch to track every
# operation on x so it can compute the derivative later

loss = x ** 2

loss.backward()  # runs the chain rule automatically through every operation
print("part 1 -- x.grad:", x.grad)  # expect 6.0

# ── part 2: chained operations ─────────────────────
# this matches exercise 4 exactly
# g(x) = 2x + 1
# loss  = g(x)**2 = h(x)
# chain rule answer from exercise 4 was 28.0

x = torch.tensor(3.0, requires_grad=True)

y    = 2 * x + 1   # g(x)
loss = y ** 2       # f(g(x))

loss.backward()     # PyTorch runs chain rule backwards automatically
print("part 2 -- x.grad:", x.grad)  # expect 28.0
print("matches exercise 4:", x.grad.item() == 28.0)

# loss.backward() is running the chain rule automatically
# across every operation in the computation graph.
# PyTorch tracked every step from x to loss and computed
# the derivative of loss with respect to x by multiplying
# the local slopes backwards through the chain.
# in a real neural network this runs across millions of
# weights simultaneously -- that is backpropagation.