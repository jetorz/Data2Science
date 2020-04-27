from sympy import Symbol, integrate, oo
import math
t = Symbol('t')

ex = integrate(t * 0.01 * math.e**(-0.01*t), (t, 0, oo))
ex.doit()

import scipy.integrate as si
import numpy as np

si.quad(lambda t: t * 0.01 * np.exp(-0.01*t), 0, np.inf)

# https://www.derivative-calculator.net/
# https://www.integral-calculator.com/
# Maxima
