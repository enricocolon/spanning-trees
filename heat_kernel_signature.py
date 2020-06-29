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
def discoverBoundary(matrix):
    border_index_pairs = []
    sgns = []
    sgnmatrix = np.sign(matrix)
    sgnmatrix[sgnmatrix == 0] = 1 #although possibly adding error, this eliminates border thickness errors.
    #collect row/col pairs
    #matrix[:,:-1] takes off last col. matrix[:,1:] takes off first col.
    #if max[i][j] neq 0, {(i,j),(i+1,j)} on border. Same principle with rows
    #for {(i,j),(i,j+1)}.
    row_border_matrix = sgnmatrix[:,:-1]-sgnmatrix[:,1:]
    row_border_matrix = np.append(row_border_matrix, np.zeros((len(row_border_matrix),1)), axis=1)
    col_border_matrix = sgnmatrix[:-1]-sgnmatrix[1:]
    col_border_matrix = np.append(col_border_matrix, np.zeros((1,len(col_border_matrix[0]))), axis=0)
    row_border_indices = np.argwhere(row_border_matrix !=0)
    col_border_indices = np.argwhere(col_border_matrix !=0)
    for row_border_index in row_border_indices:
        border_index_pairs = border_index_pairs + [((row_border_index[0],row_border_index[1]),
                                                    (row_border_index[0],row_border_index[1]+1))]
    for col_border_index in col_border_indices:
        border_index_pairs = border_index_pairs + [((col_border_index[0]+1,col_border_index[1]),
                                                    (col_border_index[0],col_border_index[1]))]
    for pair in border_index_pairs:
        sgns = sgns + [(np.sign(matrix[pair[0][0]][pair[0][1]]), np.sign(matrix[pair[1][0]][pair[1][1]]))]
    if (int(sgns[0][0]) == -1):
        bip = np.flip(border_index_pairs, axis=1)
    else:
        bip = np.array(border_index_pairs)
    return bip


test_border = discoverBoundary(z2_h_region_2(10000, 50, np.array([60,40]), np.array([53,45])))
A = np.zeros((102,102))
for a in test_border:
    A[a[1][0]][a[1][1]] = -1
    A[a[0][0]][a[0][1]] = 1


print(test_border)
sns.heatmap(A, cmap="RdBu")
plt.show()

@jit
def Q(function, z1, z2):

    return


#ax = plt.axes()
#
#sns.heatmap(z2_h_region_2(10000, 50, np.array([47,47]), np.array([53,45])), ax=ax, cmap="RdBu")
#ax.set_xlabel("x$_1$")
#ax.set_ylabel("x$_2$")
#ax.set_title("H(t,x,z)-H(t,x,z')")
#plt.show()
