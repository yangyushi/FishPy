#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.pyplot import plot, scatter, imshow
import sympy as sy
from sympy.solvers.solveset import nonlinsolve

p11, p12, p13, p14 = sy.symbols('p11 p12 p13 p14')
p21, p22, p23, p24 = sy.symbols('p21 p22 p23 p24')
p31, p32, p33, p34 = sy.symbols('p31 p32 p33 p34')

X, Y, Z = sy.symbols('X Y Z')
u, v, c = sy.symbols('u v c')

P = sy.Matrix([
    [p11, p12, p13, p14],
    [p21, p22, p23, p24],
    [p31, p32, p33, p34],
    ])


eq1 = p11 * X + p12 * Y + p13 * Z + p14 - c * u
eq2 = p21 * X + p22 * Y + p23 * Z + p24 - c * v
eq3 = p31 * X + p32 * Y + p33 * Z + p34 - c

system = [eq1, eq2, eq3]

res = nonlinsolve(system, [X, Y, c])

print(res)
