from sympy import Symbol, integrate, oo, simplify
import sympy as sp
import math
w = Symbol('w')

ex = integrate(t * 0.01 * math.e**(-0.01*t), (t, 0, oo))
ex.doit()

import scipy.integrate as si
import numpy as np

si.quad(lambda t: t * 0.01 * np.exp(-0.01*t), 0, np.inf)

# https://www.derivative-calculator.net/
# https://www.integral-calculator.com/
# Maxima

s1 = 0.25
s2 = 0.3
rho = 0.9
sp2 = w**2 * s1**2 + (1-w)**2 * s2**2 + 2*rho*w*s1*(1-w)*s2

simplify(sp2)

dw = sp.diff(sp2, w)

w1 = sp.solve(dw)[0]

u1 = 5/100
u2 = 10/100
ep = w1 * u1 + (1-w1) * u2
ep

u1 = 10/100
s1 = 10/100
u2 = 12/100
s2 = 15/100
u3 = 15/100
s3 = 18/100

rho12 = rho13 = rho23 = 0.9

sp.Matrix([[1, -1], [3, 4], [0, 2]])
v1 = sp.Matrix([1, 1, 1])
u = sp.Matrix([u1, u2, u3])
Sigma = sp.Matrix([[s1**2, rho12*s1*s2, rho13*s1*s3], [rho12*s1*s2, s2*s2, rho23*s2*s3],[rho13*s1*s3, rho23*s2*s3, s3*s3]])

a = u.T * Sigma**(-1) * u
b = u.T * Sigma**(-1) * v1
c = v1.T * Sigma**(-1) * v1
up = sp.Symbol('up')

sgm2 = c/(a*c-b**2)*up*up + (-2*b)/(a*c-b*b)*up + a/(a*c-b*b)
sgm2

rf = 2/100
p = 6/100
p05 = 100 * p / 2
p10 = 100 * p / 2
p15 = 100 * p / 2
p20 = 100 + 100 * p / 2
pv = p05 * math.exp(-rf*0.5) + p10 * math.exp(-rf * 1) + p15 * math.exp(-rf * 1.5) + p20 * math.exp(-rf*2)
pv15 = p20 * math.exp(-rf*3/12)
pv15
p20

x = [line for line in open('assets\X.txt', 'r', encoding='UTF8')]
y = [line for line in open('assets\y.txt','r', encoding='UTF8')]
data = []
for i in range(len(x)):
    data.append(x[i].strip() + y[i])
data
with open('assets\ex6data.txt', 'w', encoding='UTF8') as ex:
    ex.writelines(data)
