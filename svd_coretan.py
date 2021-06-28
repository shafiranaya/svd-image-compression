# from mpmath.libmp.libintmath import _1_100
from math import sqrt
import numpy as np
# from numpy.linalg import eig
# from numpy.linalg.linalg import eigvals
import scipy.linalg
import sympy as sp

Vt_harusnya = np.array([[-17/(5*sqrt(30)),-10/(5*sqrt(30)), 19/(5*sqrt(30))],
[-6/(5*sqrt(5)),-5/(5*sqrt(5)),  (8/(5*sqrt(5)))],
[7/(5*sqrt(6)), -2/(1*sqrt(6)), -1/(5*sqrt(6))]])
# print(-17/(5*sqrt(30)))
# print(-10/(5*sqrt(30)))
# print(19/(5*sqrt(30)))
# print()
# print(-6/(5*sqrt(5)))
# print(-6/(5*sqrt(5)))
# print(8/(5*sqrt(5)))
# print()
# print(7/(5*sqrt(6)))
# print(-2/(1*sqrt(6)))
# print(-1/(5*sqrt(6)))


def sympy_to_numpy(sympy_matrix):
    matrix_shape = sp.shape(sympy_matrix)
    numpy_matrix = np.zeros(matrix_shape)
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            numpy_matrix[i,j] = sp.N(sympy_matrix[i,j])
    return numpy_matrix

# Compute SVD

A = np.array([[3,2,1],[2,1,4]])
# A = np.array([[3,2,2],[2,3,-2]])
# A = np.array([[2,2,0],[-1,1,0]])
# A = np.array([[3,1,1],[-1,3,1]])
# print(A.shape)

# Step 1: Form A'A
Atranspose_A = np.matmul(A.transpose(),A)
print(Atranspose_A)

# Step 2: Determine eigenvalues of A'A
e_value, e_vector = np.linalg.eig(Atranspose_A)
# e_value, e_vector = scipy.linalg.eigvals(Atranspose_A)
# print("e_value = \n",e_value)

# GAKEPAKE YANG INI
# print("e_vector = \n",e_vector)
# v = np.zeros((e_vector.shape))
# for i in range(len(e_vector)):
#     v[i] = e_vector[i]
# print('v = \n', v)



eigenvalue = []
coba_eigenvector = np.linalg.eigh(Atranspose_A)
# coba_eigenvector_sort = sorted(coba_eigenvector,key=coba_eigenvector[0],reverse=True)
# coba_eigenvector.transpose()
# coba_eigenvector = []
eigenvects = sp.Matrix(Atranspose_A).eigenvects()
eigenvects = sorted(eigenvects,reverse=True) # decreasing magnitude
# print("eigenvector:\n",eigenvects)
for i in range(len(eigenvects)):
    eigenvalue.append(eigenvects[i][0])

print('eigen value: ',eigenvalue)
# print("Coba eigenvector = \n", coba_eigenvector[1].transpose())
tes_eigenvector = coba_eigenvector[1].transpose()
tes_eigenvector = np.flip(tes_eigenvector,0)
print("Tes eigenvector = \n", tes_eigenvector)
v_tes = np.zeros((e_vector.shape))
for i in range(len(e_vector)):
    v_tes[i] = tes_eigenvector[i]

# eigenvector_bismillah = [[0 for i in range(n)] for j in range (n)]
# eigenvector_bismillah = np.zeros((e_vector.shape))
# eigenvector_bismillah = []
# print("n = ",n)
# print("Eigenvector bismillah = \n", eigenvector_bismillah)
# for i in range(n):
#     eigenvector_bismillah.append()
# for row in eigenvector_bismillah:
#     for cell in row:
#         cell = coba_eigenvector[row]
# for i in range(n):
#     for j in range(n):
#         eigenvector_bismillah[i][j] = coba_eigenvector[i,j]
# print("Eigenvector bismillah = \n", eigenvector_bismillah)

# Step 3: Form the matrix V'
print('Vt harusnya:\n',Vt_harusnya)
# Vt = e_vector.transpose()
# print("Vt:\n",Vt)

# Step 4: Form the matrix Sigma (DONE)
sigma = np.zeros(A.shape)
# sigma = sigma.fill_diagonal(eigenvalue)
for i in range(len(sigma)):
    for j in range(len(sigma[0])):
        if (i == j) and (eigenvalue[i] != 0):
            sigma[i,j] = sqrt(eigenvalue[i])
print('Sigma: \n',sigma)

# Step 5: Form the matrix U
u = np.zeros((A.shape[0],A.shape[0]))
# u = []
# print(sigma.shape[0])
for i in range(sigma.shape[0]):
    # first = (1/sqrt(eigenvalue[i])) * A * v[i]
    # second
    # third
    first = np.multiply(1/(sqrt(eigenvalue[i])),A)
    # second = np.matmul(first, e_vector[i])
    # second = np.matmul(first,v[i])
    second = np.matmul(first, v_tes[i])
    u[i] = second
    # u.append(second)
print('u: \n', u)
U = u.transpose()
print("U = \n", U)
# eigenvector = np.zeros((A.shape[1],A.shape[1])) # nanti harusnya dari shape

# Step 6
# TES BENER GAK
print("A: \n",A)
A_test = np.matmul(np.matmul(U,sigma), tes_eigenvector)
print("A_test: \n",A_test)

# matrix_aneh = []
# for i in range(len(eigenvects)):
#     print(i,(eigenvects[i][2][0]))
#     matrix_aneh.append(eigenvects[i][2][0])
# print("Matriks aneh\n",matrix_aneh)

# for i in range(len(eigenvects)):
#     eigenvalue.append(eigenvects[i][0])
#     # eigenvector[i] = sp.matrices.dense.matrix2numpy(eigenvects[i][2][0])
#     # eigenvector.append(np.array((eigenvects[i][2][0]).tolist()))
#     eigenvector.append(sympy_to_numpy(eigenvects[i][2][0]))
#     # eigenvector.append(np.array(sp.matrices.dense.matrix2numpy(eigenvects[i][2][0])))
# print(eigenvalue)
# # eigenvector = np.array(eigenvector)
# print(eigenvector[0])
# print(eigenvector[1])
# print(eigenvector[2])

# print("Eigenvector = \n" ,eigenvector)
# e_value = sp.Matrix(Atranspose_A).eigenvects()
# e_value = scipy.linalg.eigvals(Atranspose_A).scipy.denorm
# e_value = eigvals(Atranspose_A)
# print("Eigen value = ", e_value)
# print("Eigen vector = ", e_value)
# for value in e_value:
#     print(value)
# print(e_value[0],e_value[1],e_value[2])
# CONTOH SOLVING HOMOGENEOUS EQUATION
# a = np.array([[-17,8,11],[8,-25,6],[11,6,-13]])
# e_value_a, e_vector_a = eig(a)
# print("Eigen value = ", e_value_a)
# print("Eigen vector = ", e_vector_a)
# a_null = sp.Matrix(a).nullspace()
# print("a null",a_null)
# b = np.array([0,0,0])
# sol = np.linalg.solve(a,b)
# print("Solution",sol)
# A = [[-12,12,2],[12,-12,-2],[2,-2,-17]]
# A_np = np.array(A)

# # reduced row

# A_reduced = sp.Matrix(A_np).rref()
# print(A_reduced[0])
# print(1/np.sqrt(2))
# lambda_25 = [[1,-1,0],[0,0,1],[0,0,0]]
# test = np.array(lambda_25)
# dinorm_x = np.linalg.norm(test,axis=0)
# print('Column wise norm: ',dinorm_x)
# normalized_x = test/dinorm_x
# print("Normalized:",normalized_x)
# dinorm_y = np.linalg.norm(test,axis=1)
# normalized_y = test/dinorm_y

# print('Row wise norm: ',dinorm_y)
# print("Normalized:",normalized_y)
