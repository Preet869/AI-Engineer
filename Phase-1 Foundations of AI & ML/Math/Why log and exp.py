"""
WHY LOG? WHY EXPONENTS? — the intuition
========================================
What it is:
Logarithms and exponents are inverse operations. Exponents grow
numbers very fast (2^10 = 1024). Logarithms shrink them back down
(log(1024) = 10). In ML both appear constantly -- often in the same
formula -- and understanding why they are chosen is more useful than
memorising the formulas.
 
Why it matters in AI:
Log is used in loss functions because it heavily punishes confident
wrong predictions. Exp is used in softmax to make all outputs
positive before normalising. Log is used to turn multiplication into
addition (easier to compute). Log is used to handle very small
probabilities that would underflow to zero in a computer. Every time
you see log or exp in an ML formula there is a specific reason for it.
 
Pipeline position:
  this is foundational -- appears inside loss functions, softmax,
  attention scores, and probability calculations everywhere
"""
 
import numpy as np

# Excerise 1 - What log does in numbers

print("=" * 50)
print("1. WHAT LOG DOES")
print("=" * 50)
 
values = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
print("value       log(value)    meaning in ML")
print("-" * 55)
for v in values:
    log_v = np.log(v)
    meaning = ""
    if v < 0.2:
        meaning = "wrong and confident -- huge penalty"
    elif v < 0.6:
        meaning = "uncertain -- moderate penalty"
    else:
        meaning = "correct and confident -- small penalty"
    print(f"  {v:<10}  {round(log_v, 4):<14}  {meaning}")
print()
print("log turns probabilities into penalties that scale correctly")
print()

# Excerise 2 - What exp does to numbers
# exp(x) = e^x where e = 2.718...
# Makes everything postitive. Amplifies differences.

print("=" * 50)
print("2. WHAT EXP DOES")
print("=" * 50)
 
logits = [-2, -1, 0, 1, 2, 3]
print("logit     exp(logit)    meaning")
print("-" * 45)
for l in logits:
    exp_l = np.exp(l)
    print(f"  {l:<10}  {round(exp_l, 4):<14}  always positive")
print()
print("exp makes all numbers positive -- needed before softmax can normalise")
print()

# Excerise 3 - log turns multiplication into addition
# this is why log is used with probabilities.
# P(A and B and C) = P(A) * P(B) * P(C)
# With many small probabilities this underflows to 0.
# log(P(A)*P(B)*P(C)) = log(P(A)) + log(P(B)) + log(P(C))
# Addition is safer for computers than tiny multiplications.

print("=" * 50)
print("3. LOG TURNS MULTIPLY INTO ADD")
print("=" * 50)
 
probs = [0.6, 0.7, 0.55, 0.8, 0.65, 0.7, 0.6, 0.75, 0.5, 0.8]
 
# multiply all probabilities -- gets tiny fast
product = np.prod(probs)
print("multiply 10 probabilities together:", product)
print("with 100 probabilities this would underflow to 0.0")
print()
 
# sum log probabilities -- stays manageable
log_sum = np.sum(np.log(probs))
print("sum of log probabilities:", round(log_sum, 4))
print("exp(log_sum) =", round(np.exp(log_sum), 6), " (same answer, stable computation)")
print()
 
# Exercise 4 - log and exp cancel each other
# exp(log(x)) = x and log(exp(x)) = x
# they are inverse operations. 

print("=" * 50)
print("4. LOG AND EXP ARE INVERSES")
print("=" * 50)
 
x = 7.3
print(f"x = {x}")
print(f"log(x) = {round(np.log(x), 4)}")
print(f"exp(log(x)) = {round(np.exp(np.log(x)), 4)}  (back to x)")
print()
print(f"exp(x) = {round(np.exp(x), 4)}")
print(f"log(exp(x)) = {round(np.log(np.exp(x)), 4)}  (back to x)")
print()

# Excerise 5 - Softmax uses exp, cross-entropy uses log
# Together they form complete output + loss calculation.

print("=" * 50)
print("5. EXP IN SOFTMAX, LOG IN CROSS-ENTROPY")
print("=" * 50)
 
logits   = np.array([2.5, 1.0, 0.2])   # raw model output
y_true   = np.array([1, 0, 0])          # true class is index 0
 
# softmax uses exp
exp_logits = np.exp(logits)
probs      = exp_logits / np.sum(exp_logits)
 
# cross-entropy uses log
loss = -np.sum(y_true * np.log(probs))
 
print("logits:", logits)
print("after exp:", exp_logits.round(4))
print("after softmax (divide by sum):", probs.round(4))
print("cross-entropy loss:", round(loss, 4))
print()
print("exp makes raw outputs into positive values")
print("softmax normalises them into probabilities")
print("log in cross-entropy punishes wrong confident predictions")
print("these three steps appear together in every classification model")