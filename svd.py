from math import sqrt
import numpy as np
from numpy.core.fromnumeric import sort
from numpy.linalg import eig
import sympy as sp
import scipy.linalg
import scipy.sparse.linalg
# cuma buat tes
import cv2
import datetime

def svd(A):
    A = np.array(A,dtype=np.int64)
    # Step 1: Form A'A
    Atranspose_A = np.matmul(A.transpose(),A)
    # Step 2: Determine eigenvalues of A'A
    eigenvalue  = np.linalg.eig(Atranspose_A)[0].real
    eigenvalue = sorted(eigenvalue,reverse=True)
    # Step 3: Form the matrix V'
    eigenvector = np.linalg.eigh(Atranspose_A)[1]
    eigenvector = eigenvector.transpose()
    Vt = np.flip(eigenvector,0)
    v = np.zeros((Vt.shape))
    for i in range(len(Vt)):
        v[i] = Vt[i]
    # Step 4: Form the matrix Sigma
    Sigma = np.zeros(A.shape)
    for i in range(len(Sigma)):
        for j in range(len(Sigma[0])):
            if (i == j) and (eigenvalue[i] > 0):
                Sigma[i,j] = sqrt((eigenvalue[i]))
    # Step 5: Form the matrix U
    u = np.zeros((A.shape[0],A.shape[0]))
    for i in range(Sigma.shape[0]):
        if (eigenvalue[i] > 0):
            u[i] = np.matmul(np.multiply(1/(sqrt((eigenvalue[i]))),A), v[i])
    U = u.transpose()
    # print("U: \n",U)
    # print("Sigma: \n",Sigma)
    # print("Vt: \n",Vt)

    # # BANDINGIN SAMA HASIL LIBRARY
    # U_lib, S_lib, Vt_lib = np.linalg.svd(A)
    # print("U_lib: \n",U_lib)
    # S_lib = sorted(S_lib, reverse=True)
    # print("S_lib: \n",S_lib)
    # print("Vt_lib: \n",Vt_lib)
    return U, Sigma, Vt

### DRIVER ###
# # # Compute SVD
# img_name = 'momo_kecil_gray.png'
# path = 'in/' + img_name
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # # # print(img)
# # # print(gray)
# print("img shape:", img.shape)
# print("gray shape:" ,gray.shape)
# # start = datetime.datetime.now()

# A = gray
# A = np.array(A,dtype=np.int32)

# # A = np.random.randint(255, size=(3,5))
# # A = np.array([[33,222,111],[222,111,44]])
# A = np.array([[3,2,1],[2,1,4]])

# print(A)
# print(A.shape)
# new = svd(A)
# print(new)
# end = datetime.datetime.now()
# path_out = 'out/' + img_name + '.png'
# cv2.imwrite(path_out,new)
# print((end-start))
# Percobaan 1: 4783


# print(svd(gray))
# # Step 6
# # TESTING BENER APA ENGGA
# print("---TESTING---")
# print("A: \n",A)
# U, Sigma, Vt = svd(A)
# A_test = np.matmul(np.matmul(U,Sigma), Vt)
# print("A_test: \n",A_test)
# print("U: \n",U)
# print("Sigma: \n",Sigma)
# print("Vt: \n",Vt)

# # BANDINGIN SAMA HASIL LIBRARY
# U_lib, S_lib, Vt_lib = np.linalg.svd(A)
# print("U_lib: \n",U_lib)
# S_lib = sorted(S_lib, reverse=True)
# print("S_lib: \n",S_lib)
# print("Vt_lib: \n",Vt_lib)