from math import *
import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import itertools
import random
import fractions

from graders309.exceptions import *
from graders309.helpers import *

from functools import reduce


def random_int_matrix(m=3, n=3, matrix_min=-3, matrix_max=4, invertible=False, max_det=None):
    '''
    Produces a random mxn matrix
    with integer coefficients

    :m:
        Number of rows of the matrix.
    :n:
        Number of columns of the matrix.
    :matrix_min:
        Minimum possible value of entries of the matrix.
    :matrix_max:
        Maximum possible value of entries of the matrix.
    :invertible:
        Boolean, if True the matrix will be invertible
        This is used only is m==n.
    :max_det:
        Positive integer. The maximum possible value of the determinant
        of the matrix. This is used only if m==n and invertible==True.

    Returns:
        A 2-dimensional numpy array of integers.
    '''

    A = np.random.randint(matrix_min, matrix_max + 1, (m,n))

    if m==n and invertible:
        if max_det is not None and max_det <= 0:
            max_det = None
        while True:
            abs_det = abs(np.linalg.det(A))
            if abs_det > 0.5 and (True if max_det is None else abs_det <= abs(max_det)):
                break
            else:
                A = np.random.randint(matrix_min, matrix_max + 1, (m,n))
    return  A


def random_int_combination(A, coeff_min = -6, coeff_max = 7, nonzero=False):
    '''
    For a given matrix A returns a randomly selected linear combination
    of columns of A with integer coefficients. The returned values is a tuple,
    (v, c) where v is the linear compbination vector, and c is a list of
    scalars which represent v as a linear combination of vectors in B.

    :A:
        A numpy array representing the matrix.
    :coeff_min:
    :coeff_max:
        The minimium and maximum value of possible coefficients
        of the combination.
    :nonzero:
        Boolean. If True all coefficient of the linear combination will be
        non-zero.

    Returns:
        a tuple, (v, c) where v is a numpy array with the linear compbination
        vector, and c is a numpy array with coefficients of the combination.
    '''

    population = list(range(coeff_min, coeff_max + 1))
    if nonzero and 0 in population:
        population.remove(0)
    #UBx Python libraries do not have np.random choice or random.choices
    #so we need to improvise...
    c = np.array([random.choice(population) for i in range(A.shape[1])])
    v = A.dot(c)
    return (v, c)



def fixed_rank_random_matrix(m, n, rank=None, matrix_min=-3, matrix_max=3, max_det=None, max_denominator=None):

    '''
    Returns a  randomly generated matrix A with integer coefficients of a given
    dimensions and rank. The matrix A is obtained as a product A = LUR where
    L and R are invertible matrices, and U is a matrix with an approriate number
    of 1 entries on the main diagonal, and all other entries equal to 0.

    :m:
        Number of rows of the matrix A.
    :n:
        Number of columns of the matrix A.
    :rank:
        Rank of the matrix A. If None, a matrix of the largest possible rank
        will be returned.
    :matrix_min:
        The minimum possible value of entries of the matrices L and R.
        This, together with the arguments matrix_min and max_det can
        help control the magnitude of entries of the matrix A.
    :matrix_max:
        The maximum possible value of entries of the matrices L and R.
    :max_det:
        Positive integer. The maximum possible value of determinants
        of matrices L and R.
    :max_denominator:
        Positive integer. The maximum possible value of denominators of fractions
        in entries of the row reduced echelon form of A.

    Returns:
        A 2-dimensional numpy array of integers.
    '''


    if (rank is None) or (rank > min(m, n)):
        rank = min(m, n)

    if rank < 0:
        rank = 0

    while True:
        L = random_int_matrix(m = m,
                             n = m,
                             matrix_min = matrix_min,
                             matrix_max = matrix_max,
                             invertible=True,
                             max_det=max_det)

        R = random_int_matrix(m=n,
                             n=n,
                             matrix_min = matrix_min,
                             matrix_max = matrix_max,
                             invertible=True,
                             max_det=max_det)

        U = np.zeros((m, n), dtype=int)

        U[range(rank), range(rank)] = 1
        A = np.dot(L, np.dot(U, R))
        RA = sympy.Matrix(A).rref()[0]
        RA_max_denom = max([sympy.fraction(x)[1] for x in RA])

        if max_denominator is not None:
            if RA_max_denom <= max_denominator:
                break
        else:
            break
    return A



def random_pivots_matrix(m=4, n=7, rank=3, coeff_min=-3, coeff_max = 3, nonzero=False):

    """
    Generated a random mxn matrix A with integer coefficients of a given rank
    and with randomly selected pivot columns.

    :m:
        Number of rows of the matrix A.
    :n:
        Number of columns of the matrix A.
    :rank:
        Rank of the matrix A.
    :coeff_min:
    :coeff_max:
        Integers. These coefficients  control magnitude of entries in the
        matrix A. It is required that coeff_min < coeff_max.
    :nonzero:
        Boolean. Non-pivot columns of the matrix A are generated as linear
        combinations of pivot columns. If this argument is True all coefficient
        of these linear combinations will be non-zero.
    Returns:
        A tuple (A, pivots) where A is a numpy array representing the matrix
        and pivots is a list of numbers of pivot columns of A in the increasing
        order.
    """

    if rank > min(m, n):
        rank = min(m, n)

    pivots = sorted(random.sample(range(n), rank))
    pivots[0] = 0
    PP = np.eye(m, m, dtype=int)[:, :rank]
    P = np.zeros((m, n), dtype=int)
    P[range(rank), pivots] = 1


    # piv_num is the index of the fist pivot greater then the current column number
    piv_num = 0
    # iterate over non-pivot columns
    for i in [x for x in range(n) if x not in pivots]:
        # check if piv_num needs to be increased:
        while piv_num < len(pivots) and i > pivots[piv_num] :
            piv_num += 1
        # compute column i of the matrix P as a linear combination
        # of pivot columns preceding this column
        v = random_int_combination(PP[:, :piv_num],
                                   coeff_min = coeff_min,
                                   coeff_max = coeff_max,
                                   nonzero=nonzero)[0]
        P[:, i] = v

    Q = random_int_matrix(m=m, n=m, matrix_min= coeff_min , matrix_max=coeff_max, invertible=True)
    A = np.dot(Q, P)

    return A, pivots



def random_orthogonal_basis(n, matrix_min=-2, matrix_max = 2, max_denominator=None, int_coeff=False):
    """
    Generates an orthogonal basis of R^n by applying
    G-S process to a random basis of R^n with integer coefficients.

    Note: if max_denominator is set to a small value, this function
    is reasonably fast for small values of n only (n <=4).

    :n:
        The dimension of R^n
    :matrix_min:
        Minimum possible value of entries of the basis before G-S process is applied.
    :matrix_max:
        Maximum possible value of entries of the basis before G-S process is applied.
    :max_denominator:
        After G-S process the orthogonal basis will in general have fractions as its
        coordinates. This argument sets that maximal allowed value of denominators of
        fractions appearing in the orthogonal basis.
    :int_coeff:
        Boolean. If true the orthogonal basis will be modified so that coordinated of
        all its vectors are integers. This is accomplished by multiplying each vector
        obtained in the G-S process by lcm of denominators of its entries, and then
        dividing the resulting vector of integers by gcd of its entries.
    Returns:
        A tuple (A, GS) where A is the nxn numpy array whose columns give the basis
        of R^n to which G-S process was applied, and GS is a list of sympy vectrors
        of the resulting orthogonal basis.
    """
    while True:
        # start with a random basis of R^n
        A = random_int_matrix(m=n,
                                 n=n,
                                 matrix_min=matrix_min,
                                 matrix_max=2,
                                 invertible=True)


        # get an orthogonal basis using the G-S process
        GS = sympy.GramSchmidt([sympy.Matrix(A[:,i]) for i in range(n)])

        # collect the list of denominators of fractions appearing
        # in the orthogonal basis
        d_list = []
        for w in GS:
            denoms = [sympy.fraction(x)[1] for x in w]
            d_list.append(denoms)


        # if the max_denominator is set check if the orthogonal basis
        # satisfies this requirement
        if max_denominator is None:
            break
        max_d = max([max(x) for x in d_list])
        if max_d <= max_denominator:
            break


    # if int_coeff == True convert the orthogonal basis into
    # one with integral coefficients, by multiplying each basis
    # vector by lcm of denominators of entires of the vector
    if int_coeff:
        def lcm(a, b):
            return a*b/fractions.gcd(a, b)

        lcm_list =  [reduce(lcm, ds) for ds in d_list]
        GS = [c*w for (c, w) in zip(lcm_list, GS)]

        # to further simplify the orthogonal integer basis,
        # divide each basis vector by gcd of its entries
        for i in range(n):
            w_gcd =  reduce(fractions.gcd, [int(x) for x in GS[i]])
            GS[i] = GS[i]/abs(w_gcd)

    return A, GS



class WebPages:
    """
    Generated a network of webpages and its plot.
    Computes simplified PageRanks of nodes in the network.
    """

    def __init__(self, N=7, incidence=None, edges=None, simple=True):
        """
        :N:
            The number of nodes in the network.
        :incidence:
            NxN incidence matrix of the network. If none the incidence
            matrix will be generated randomly. (See also TO DO below)
        :edges:
            The number of edges in the network. Used only if the incidence
            matrix is randomly generated. If simple == True the number of
            edges in the generated network may exceed this number in order
            to assure that the network is strongly connected.
        :simple:
            Only used when the network is randomly generated. If True the
            generated network will be strongly connected, i.e. it will be
            possible to get from any vertex to any other vertex along directed
            edges.

        TO DO:
           - to adhere with the common terminology the "incidence matrix" should
             be transposed and should be called the adjacency matrix.
        """

        self.N = N

        if edges == None:
            edges = self.N
        else:
            edges = min(edges, self.N*(self.N -1))

        if incidence is not None:
            self.incidence = incidence
        elif simple:
            self.incidence = self.random_edges_simple(edges)
        else:
            self.incidence = self.random_edges(edges)

    def random_tree(self):
        """
        Randomly generates a directed tree.
        """

        matrix = np.zeros((self.N, self.N), dtype=int)
        source = list(range(self.N))
        random.shuffle(source)
        root = source.pop()
        dest = [root]
        while source:
            d = random.choice(dest)
            s = source.pop()
            matrix[d, s] = 1
            dest.append(s)
        return matrix, root


    def random_edges_simple(self, edges):
        """
        Randomly generates a strongly connected network. This is
        done by generating a tree then adding edges as needed.
        """

        matrix, root = self.random_tree()

        roots = {root}
        while True:
            A = matrix.copy()
            for k in range(self.N):
                A  += np.dot(A, matrix)

            leaves = list(np.flatnonzero(np.sum(A, axis=1) == 0))
            if len(leaves) == 0:
                break
            i = random.choice(leaves)
            j = random.choice(list(roots))
            matrix[i, j] = 1
            new_roots = set(np.flatnonzero(A[:,i]))
            roots.update(new_roots)



        diff = edges - np.sum(matrix)
        if diff > 0:
            matrix = matrix.ravel()
            reye = np.eye(self.N, dtype=int).ravel()
            rmatrix = matrix + reye
            zeros_idx = np.arange(self.N**2, dtype=int)
            zeros_idx = zeros_idx[rmatrix == 0]
            s = random.sample(list(zeros_idx), diff)
            matrix[s] = 1
            matrix = matrix.reshape(self.N, self.N)

        return matrix


    def random_edges(self, edges):
        """
        Randomly generates a  network. The resulting network
        need not be connected or strongly connected.
        """

        matrix = np.zeros(self.N**2, dtype=int)
        eye = np.eye(self.N, dtype=int).ravel()
        off_diagonal = np.arange(self.N**2, dtype=int)[eye == 0]
        indices = random.sample(list(off_diagonal), edges)
        matrix[indices] = 1
        matrix = matrix.reshape((self.N, self.N))
        matrix[range(self.N), range(self.N)] = 0
        return matrix

    def simple_page_rank(self):
        """
        Computes simplified PageRanks of network nodes.
        """

        matrix = self.incidence.T
        coeffs = matrix/np.sum(matrix, axis=0) - np.eye(self.N)
        coeffs = np.vstack((coeffs, np.ones(self.N)))
        b = np.zeros(self.N+1)
        b[-1] = 1
        ranks = np.linalg.lstsq(coeffs, b)[0]
        return [float(r) for r in ranks]


    def plot_web(self, figsize=(7,7), show=False):
        """
        Creates a plot of the network
        """

        # spacing is the radius of the circle containing centers of nodes
        spacing = self.N*0.7
        #limit of the plot
        lim = spacing + 2
        node_color = (1.00,0.86,0.15)
        edge_color = "steelblue"
        arrow_offset = 1.4   #controls placement of arrow tips
        head_width=0.3       # arrowhead width
        head_length=0.35     # arrowhead length
        radius = 1           # radius of node circles

        a = np.linspace(0, 2*np.pi, self.N+1)[:-1]
        v = spacing*np.vstack((np.sin(a), np.cos(a))).T

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        nodes = []
        for i in range(self.N):
            circle_patch = patches.Circle(v[i], radius = radius)
            nodes.append(circle_patch)

        node_collection = PatchCollection(
            nodes,
            facecolor= node_color,
            edgecolor='k'
        )

        ax.add_collection(node_collection)
        for i in range(self.N):
            plt.text(*v[i],
                     s = "{}".format(i+1),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=int(150/self.N),
                     color = 'k'
            )

        for i, j in itertools.product(range(self.N), range(self.N)):
            if  not self.incidence[i, j]:
                continue
            d = v[j]-v[i]
            norm_d = np.linalg.norm(d)
            disp = ((norm_d - arrow_offset)/norm_d)*d
            ax.arrow(
                v[i,0], v[i, 1],
                disp[0], disp[1],
                fc = edge_color,
                ec = edge_color,
                head_width = head_width,
                head_length = head_length,
                lw = 4,
                zorder = 0
            )

        plt.tight_layout(pad=0)
        if show:
            plt.show()
        enc = plot2img(fig)
        return enc



def check_span(A, B, tol=10**(-8)):

    '''
    Checks if the span of columns of a matrix B is the same
    as the span of columns of a matrix A.

    :A:
        A numpy array representing a matrix.

    :B:
        A numpy array representing a matrix.

    :tol:
        A float. Tolerance used to compute ranks of matrices. If
        None, the default value is used.

    Returns:
        True if spans of columns of A and B coincide, False otherwise.
    '''

    if len(A.shape) == 1:
        A = A.reshape(-1, 1)
    if len(B.shape) == 1:
        B = B.reshape(-1, 1)

    r_A = A.shape[0]
    r_B = B.shape[0]


    # check in A and B have the same number of rows
    if r_A != r_B:
        return False

    AB = np.hstack((A,B))
    BA = np.hstack((B,A))

    # is spans of columns of A and B coincide then rank AB will
    # be the same as A, and BA will have the same rank as B
    if (np.linalg.matrix_rank(AB, tol=tol) == np.linalg.matrix_rank(A, tol=tol)
        and np.linalg.matrix_rank(BA, tol=tol) == np.linalg.matrix_rank(B, tol=tol)
       ):
        return True
    else:
        return False


def check_lin_indep(A, tol=10**(-8)):

    '''
    Checks if columns of a matrix A are linearly independent

    :A:
        A numpy array representing a matrix.

    :tol:
        A float. Tolerance used to compute ranks of matrices. If
        None, the default value is used.

    Returns:
        True if columns of A are linearly independent, False otherwise.
    '''



    #number of columns of A
    c_A = A.shape[1]

    # columns of A are linearly independent if rank A is equal to the
    # number of columns of A
    return c_A == np.linalg.matrix_rank(A, tol=tol)


def check_basis(A, B, tol=10**(-8)):

    '''
    Checks if columns of a matrix B form a basis of the column space
    of a matrix A.

    :A:
        A numpy array representing a matrix.

    :B:
        A numpy array representing a matrix.
    :tol:
        A float. Tolerance used to compute ranks of matrices. If
        None, the default value is used.

    Returns:
        True if columns of B give a basis of Col(A), False otherwise.
    '''


    # check if Col(A) = Col(B)
    if not check_span(A, B):
        return False

    # check if columns of B are linearly independent.
    if not check_lin_indep(B):
        return False

    return True



def check_orthogonal_set(A, atol=10**(-8), rtol=0):
    '''
    Checks if columns of a matrix A form an orthogonal set of vectors

    :A:
        A numpy array representing a matrix.
    :atol:
    :rtol:
        Floats. Absolute and relative tolerances used to identify zero
        matrices.

    Returns:
        True if columns of A form an orthogonal set, False otherwise.
    '''
    B = np.dot(A.T, A)
    C = B - np.diag(np.diagonal(B))
    return np.allclose(C, np.zeros_like(C), atol=atol, rtol=rtol)
