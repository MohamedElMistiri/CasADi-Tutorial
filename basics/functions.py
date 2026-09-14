'''
In this file, we go through the basics of CasADi and how to define functions, obtain symbolic expressions, and evaluate them. We also show how to use CasADi's automatic differentiation capabilities to compute derivatives of functions.
'''
# %% import dependencies
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
# %% define SX symbolic variables and mathematical functions 
x = SX.sym('x')
y = SX.sym('y')
print(x, y)
f = vertcat(sin(x**2 + y), x**2 + y**2) # define a vector-valued function
print(f)
print(jacobian(f, x)) # obtain the partial derivative of f with respect to x
print(jacobian(f, y)) # obtain the partial derivative of f with respect to y
# print(hessian(f, x)) # obtain the second derivative of f with respect to x
# print(hessian(f, y)) # obtain the second derivative of f with respect to y
# in one step 
print(jacobian(f, vertcat(x, y))) # obtain the partial derivatives of f with respect to x and y in one step

# obtain sparcity pattern of the Jacobian
J_x = jacobian(f, x)
print(J_x.sparsity())

# %% the same can be done with MX symbolic variables, which are more efficient for large-scale problems
x = MX.sym('x')
y = MX.sym('y')
print(x, y)
f = sin(x**2 + y)
print(f)
print(jacobian(f, x)) # obtain the partial derivative of f with respect to x
print(jacobian(f, y)) # obtain the partial derivative of f with respect to y

# %% define a symbolic matrix and a function that takes a matrix as input
A = SX.sym('A', 3, 2)
x = SX.sym('x', 2)
f = A @ x
print(f)
print(jacobian(f, A)) # obtain the partial derivative of f with respect to A
print(jacobian(f, x)) # obtain the partial derivative of f with respect to x

# %% now we move on to callable casadi functions, which can be evaluated at specific numerical values
x = SX.sym('x')
y = SX.sym('y')
f = sin(x**2 + y)
f_func = Function('f_func', [x, y],
                   [vertcat(sin(x**2 + y), x**2 + y**2)],
                   ['x', 'y'],
                   ['f(x, y)']) # define a casadi function that takes x and y as input and returns f
print(f_func) # print the function
print(f_func(1, 2)) # evaluate the function at x=1 and y=2
print(jacobian(f_func(x, y), x)) # obtain the partial derivative of f_func with respect to x
print(f_func(x,2))

# %%
