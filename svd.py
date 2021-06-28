from math import sqrt
import numpy as np
import sympy as sp

# Compute SVD
A = np.array([[3,2,1],[2,1,4]])

# def svd(A):

# Step 1: Form A'A
Atranspose_A = np.matmul(A.transpose(),A)
# print(Atranspose_A)

# Step 2: Determine eigenvalues of A'A
eigenvalue = []
eigenvector = np.linalg.eigh(Atranspose_A)[1]
eigenvects = sp.Matrix(Atranspose_A).eigenvects()
eigenvects = sorted(eigenvects,reverse=True) # decreasing magnitude
for i in range(len(eigenvects)):
    eigenvalue.append(eigenvects[i][0])
# print('Eigenvalue: ',eigenvalue)

# Step 3: Form the matrix V'
eigenvector = eigenvector.transpose()
Vt = np.flip(eigenvector,0)
print("Vt = \n", Vt)
v = np.zeros((Vt.shape))
for i in range(len(Vt)):
    v[i] = Vt[i]

# Step 4: Form the matrix Sigma
sigma = np.zeros(A.shape)
for i in range(len(sigma)):
    for j in range(len(sigma[0])):
        if (i == j) and (eigenvalue[i] != 0):
            sigma[i,j] = sqrt(eigenvalue[i])
print('Sigma: \n',sigma)

# Step 5: Form the matrix U
u = np.zeros((A.shape[0],A.shape[0]))
for i in range(sigma.shape[0]):
    u[i] = np.matmul(np.multiply(1/(sqrt(eigenvalue[i])),A), v[i])
print('u: \n', u)
U = u.transpose()
print("U = \n", U)

# Step 6
# TESTING BENER APA ENGGA
print("---TESTING---")
print("A: \n",A)
A_test = np.matmul(np.matmul(U,sigma), Vt)
print("A_test: \n",A_test)
print("U: \n",U)
print("sigma: \n",sigma)
print("Vt: \n",Vt)

# BANDINGIN SAMA HASIL LIBRARY
U_lib, S_lib, Vt_lib = np.linalg.svd(A)
print("U_lib: \n",U_lib)
print("S_lib: \n",S_lib)
print("Vt_lib: \n",Vt_lib)