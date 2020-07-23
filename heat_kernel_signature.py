import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, cuda
from collections import Counter
import scipy


@jit
def integrand(x, a, t, dim):
    return np.exp(-2*(t/dim)*np.sin(x)**2)*np.cos(2*a*x)


@jit
def h_lattice(time, x_vert, y_vert):
    dim = len(x_vert)
    diff = [np.absolute(x_vert[0]-y_vert[0]), np.absolute(x_vert[1]-y_vert[1])]
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


#if directed, arrows point for edge (u,v) u ---> v
class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

@jit
def getIncidence(graph):
    matrix = np.zeros((len(graph.vertices), len(graph.edges)), dtype=int)
    for i in range(0, len(graph.edges)):
        matrix[graph.vertices.index(graph.edges[i][0])][i] = 1
        matrix[graph.vertices.index(graph.edges[i][1])][i] = -1
    return matrix


@jit
def getLaplacian(graph):
    incidence = np.array(getIncidence(graph))
    matrix = np.dot(incidence, incidence.T)
    return matrix


@jit
def getT_matrix_neghalf(graph):
    #flat refers to flat array of vertices. reshaping and converting to hashable dtype
    T = np.zeros((len(graph.vertices), len(graph.vertices)))
    flat = np.ndarray.flatten(np.array(graph.edges))
    flat = np.reshape(flat, (int(len(flat)/2),2))
    flat_array = []
    for i in flat:
        flat_array = flat_array + [tuple(i)]
    flat_array = tuple(flat_array)
    counter = Counter(flat_array)
    for i in range(0, len(graph.vertices)):
        T[i][i] = (counter[graph.vertices[i]]**(-.5))
    return T


@jit
def getScriptLaplacian(graph):
    LTneghalf = np.dot(getLaplacian(graph), getT_matrix_neghalf(graph))
    scriptL = np.dot(getT_matrix_neghalf(graph), LTneghalf)
    return scriptL


def heat_kernel_S_function(subgraph, x, y, t):
    scriptLaplacian = getScriptLaplacian(subgraph)
    def heat_kernel(s):
        h_of_t = linalg.expm(-1*(t-s)*scriptLaplacian)[subgraph.vertices.index(x)][subgraph.vertices.index(y)]
        return h_of_t
    return heat_kernel



def getDsZ2(subgraph):
    laplacian = getLaplacian(subgraph)
    z_set = []
    possible_ds = np.empty(0)
    for i in range(0, len(laplacian)):
        if laplacian[i][i] != 4:
            z_set = z_set + [subgraph.vertices[i]]
    for i in z_set:
        neigh_edges = np.array([
        (i, (i[0]+1, i[1])),
        (i, (i[0]-1, i[1])),
        (i, (i[0], i[1]+1)),
        (i, (i[0], i[1]-1))
        ])
        possible_ds = np.append(possible_ds, neigh_edges)
    possible_ds = possible_ds.reshape(int(len(possible_ds)/4), 4)
    possible_ds = np.array(possible_ds, dtype=int)
    edges_bothDirs = np.array([np.array(subgraph.edges), np.flip(np.array(subgraph.edges), axis=1)])
    edges_bothDirs = np.ndarray.flatten(edges_bothDirs)
    edges_bothDirs = edges_bothDirs.reshape(int(len(edges_bothDirs)/4), 4)
    #make set of all edges. count both directions. intersect and then setminus
    nrows, ncols = possible_ds.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [possible_ds.dtype]}
    ds = np.setdiff1d(possible_ds.view(dtype), edges_bothDirs.view(dtype))
    ds = [((i[0], i[1]),(i[2], i[3])) for i in ds]
    return ds


#input = graph, tuple #output = dictionary
def  z2_h_differences(subgraph, y):
    Ds = getDsZ2(subgraph)
    def diff(t):
        diff_over_Ds = []
        diff_over_Ds = {
        edge: h_lattice(t, edge[0], y) - h_lattice(t, edge[1], y) for edge in Ds
        }
        return diff_over_Ds
    return diff

def  z2_h_diff(subgraph, z, zprime, y):
    def diff(s):
        diff_wrt_s = h_lattice(s, z, y) - h_lattice(s, zprime, y)
        return diff_wrt_s
    return diff


def z2_h_region(time, viewWindow):
    t = time
    h_vals = np.empty((viewWindow[0], viewWindow[1]))
    for i in range(0, viewWindow[0]):
        for j in range(0, viewWindow[1]):
            h_vals[i][j] = h_lattice(t, np.array([np.floor(viewWindow[0]/2), np.floor(viewWindow[1]/2)]), np.array([i, j]))
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


#indexes increasing y, then increasing x.
@jit
def getRectS_Z2(lowerLeft, upperRight):
    if upperRight[0] < lowerLeft[0]:
        return None
    elif upperRight[1] < lowerLeft[1]:
        return None
    vertices = []
    edges = []
    width = np.abs(upperRight[0]-lowerLeft[0])
    height = np.abs(upperRight[1]-lowerLeft[1])
    for i in range(0, width+1):
        for j in range(0, height+1):
            vertices = vertices + [tuple(np.array(lowerLeft)+np.array([i,j]))]
    for i in range(0, width):
        for j in range(0, height):
            edges = edges + [(((i,j),(i+1,j)))]+[(((i,j),(i,j+1)))]
    for i in range(0, width):
        edges = edges + [((i, height),(i+1, height))]
    for j in range(0, height):
        edges = edges + [((width, j),(width, j+1))]
    S = Graph(vertices, edges)
    return S



@jit
def getRectangleSubgraph(lowerLeft, upperRight):
    if upperRight[0] < lowerLeft[0]:
        return None
    elif upperRight[1] < lowerLeft[1]:
        return None
    vertices = [(i,j) for i in range(lowerLeft[0], upperRight[0]+1)
                      for j in range(lowerLeft[1], upperRight[1]+1)]
    edges = [((v[0], v[1]),(v[0]+1,v[1])) for v in vertices] + [((v[0], v[1]),(v[0],v[1]+1)) for v in vertices]
    edges_false_top = [((i,upperRight[1]),(i,upperRight[1]+1)) for i in range(lowerLeft[0], upperRight[0]+1)]
    edges_false_bottom = [((upperRight[0],j),(upperRight[0]+1,j)) for j in range(lowerLeft[1], upperRight[1]+1)]
    edges_false = edges_false_top + edges_false_bottom
    for i in edges_false:
        edges.remove(i)
    subgrf = Graph(vertices, edges)
    return subgrf


#test_graph_left = getRectS_Z2((-3,-4),(-1,4))
#test_graph_right = getRectS_Z2((2,-4),(4,4))
#missed_vertices = [(i,j) for i in [0,1] for j in [2,3,4,-2,-3,-4]]
#missed_edges = [((i,j),(i+1,j)) for i in [-1, 0, 1] for j in [2,3,4,-2,-3,-4]]
#test_graph = Graph(test_graph_left.vertices + test_graph_right.vertices + missed_vertices + [(0,0)],
#                   test_graph_left.edges + test_graph_right.edges + missed_edges + [((-1,0),(0,0))])


@jit
def getWorstCaseSubgraph(lowerLeft, upperRight):
    graph = getRectangleSubgraph(lowerLeft, upperRight)
    newVertices = graph.vertices
    newEdges = graph.edges
    edgesRemove = [
        ((-1,1),(0,1)),((-1,-1),(0,-1)),
        ((0,-2),(0,-1)),((1,-2),(1,-1)),
    ]
    tempVertices = [(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
    edgesRemove = edgesRemove + [(i,(i[0],i[1]+1)) for i in tempVertices] + [(i,(i[0]+1,i[1])) for i in tempVertices]
    for i in [(0,1), (0,-1), (1,1), (1,-1), (1,0)]:
        newVertices.remove(i)
    for j in edgesRemove:
        newEdges.remove(j)
    wcs = Graph(newVertices, newEdges)
    return wcs


def shortWCS(n):
    A = getWorstCaseSubgraph((1-n,-n),(n,n))
    return A

#print(z2_h_differences(getWorstCaseSubgraph((-2,-2),(2,2)), (0,0))(1))
##print(integrate.quad(heat_kernel_S_function(getWorstCaseSubgraph((-2,-2),(2,2)), (0,0), (0,0), 1), 0, 1))
#print(z2_h_diff(getWorstCaseSubgraph((-2,-2),(2,2)), (0,0), (1,0), (0,0))(2))



def Qh(subgraph, t, x, y):
    zzprimes = getDsZ2(subgraph)
    sum = 0
    for pairs in zzprimes:
        def integrand(s):
            h_funct = heat_kernel_S_function(subgraph, x, y, t)
            diff_funct = z2_h_diff(subgraph, pairs[0], pairs[1], y)
            return h_funct(s)*diff_funct(s)
        sum = sum + integrate.quad(integrand, 0, t)[0]
    sum = .25*sum
    return sum


#consider Qh(t, (0,0), (-1,0)) over our potential worst case graph, where the
#argument of shortWCS is the "thickness of the donut" (n-2 edges from inside donut to out)
print(Qh(shortWCS(7), 1, (0,0), (-1,0)))


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


#test_border = discoverBoundary(z2_h_region_2(10000, 50, np.array([60,40]), np.array([53,45])))
#A = np.zeros((102,102))
#for a in test_border:
#    A[a[1][0]][a[1][1]] = -1
#    A[a[0][0]][a[0][1]] = 1

#print(integrate.quad(matExp, 0, 1))
#WIP
@jit
def getS_Z2(vertices):
    edges = []
    for i in vertices:
        for j in [np.array((0, 1)), np.array((0, -1)), np.array((1,0)), np.array((-1, 0))]:
            #if (np.array(i)+j)
            edges = edges + [(i, tuple(np.array(i)+j))]
    for e in range(0, len(edges)):
        if (edges[e][1], edges[e][0]) in edges:
            edges.remove((edges[e][1], edges[e][0]))
    return edges


#bullshit to test array of H differences


#print(getScriptLaplacian(getRectS_Z2((0,0),(3,2))))
#print(getLaplacian(Graph([1,2,3,4,5],[(1,2),(5,2)])))


#print(integrate.quad(lambda t: matExpInt([[2,-1,-1],[-1,2,-1],[-1,-1,2]], [0,0], t), 0, 1))

#ax = plt.axes()

#sns.heatmap(z2_h_region(1000,np.array([101, 101])), ax=ax)
#ax.set_xlabel("x$_1$")
#ax.set_ylabel("x$_2$")
#ax.set_title("H(t,x,y)")
#plt.show()
