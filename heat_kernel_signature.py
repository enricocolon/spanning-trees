import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, cuda


@jit
def integrand(x, a, t, dim):
    return np.exp(-2*(t/dim)*np.sin(x)**2)*np.cos(2*a*x)


@jit
def h_lattice(time, x_vert, y_vert):
    dim = len(x_vert)
    diff = np.absolute(x_vert-y_vert)
    h_inf_paths = np.empty(0)
    h_tot = 1
    t = time
    for i in range(0, dim):
        delta = diff[i]
        h_int = 2/np.pi*integrate.quad(integrand, 0, np.pi/2, args=(delta, t, len(x_vert)))[0]
        h_inf_paths = np.append(h_inf_paths, h_int)
        h_tot = h_tot*h_int
    return h_tot
#returns H given t,x,y in R^2



def z2_h_region(time, viewWindow):
    t = time
    h_vals = np.empty((viewWindow[0], viewWindow[1]))
    for i in range(0, viewWindow[0]):
        for j in range(0, viewWindow[1]):
            h_vals[i][j] = h_lattice(t, np.array([0, 0]), np.array([i, j]))
    return h_vals
#returns values for H(t,(0,0),y) for some window size.


@jit
def z2_h_region_2(time, radius, z, zprime):
    t = time
    h_vals = np.empty((2*radius+1, 2*radius+1))
    for i in range(0, 2*radius+1):
        for j in range(0, 2*radius+1):
            h_vals[i][j] = h_lattice(t, np.array([z[0], z[1]]), np.array([i, j]))-h_lattice(t, np.array([zprime[0], zprime[1]]), np.array([i, j]))
    return h_vals
#returns H(t,x,z)-H(t,x,z') for all x in [0, 2rad+1]^2 and fixed z, z'.


@jit
def corrector(array, n):
    corrected = (2/np.pi)*np.arctan(n*array)
    return corrected
#the corrector allows numbers close to zero to become more visible on a heatmap.


@jit
def Q(function, z1, z2):

    return


ax = plt.axes()

sns.heatmap(z2_h_region_2(10000, 50, np.array([47,47]), np.array([53,45])), ax=ax, cmap="RdBu")
ax.set_xlabel("x$_1$")
ax.set_ylabel("x$_2$")
ax.set_title("H(t,x,z)-H(t,x,z')")
plt.show()
