from math import sqrt
import numpy as np
import sympy as sp
# cuma buat tes
import cv2
import datetime

def svd(A):
    # Step 1: Form A'A
    Atranspose_A = np.matmul(A.transpose(),A)
    # print(Atranspose_A)

    # Step 2: Determine eigenvalues of A'A
    eigenvalue = []
    # eigenvects = sp.Matrix(Atranspose_A).eigenvects()
    # eigenvects = sorted(eigenvects,reverse=True) # decreasing magnitude
    # for i in range(len(eigenvects)):
    #     eigenvalue.append(eigenvects[i][0])
    eigenvects = sorted(sp.Matrix(Atranspose_A).eigenvects(),reverse=True) # decreasing magnitude
    for i in range(len(eigenvects)):
        eigenvalue.append(eigenvects[i][0])
    # print('Eigenvalue: ',eigenvalue)

    # Step 3: Form the matrix V'
    # eigenvector = np.linalg.eigh(Atranspose_A)[1]
    # eigenvector = eigenvector.transpose()
    # Vt = np.flip(eigenvector,0)
    # # print("Vt = \n", Vt)
    # v = np.zeros((Vt.shape))
    # for i in range(len(Vt)):
    #     v[i] = Vt[i]
    Vt = np.flip(np.linalg.eigh(Atranspose_A)[1].transpose(),0)
    # print("Vt = \n", Vt)
    v = np.zeros((Vt.shape))
    for i in range(len(Vt)):
        v[i] = Vt[i]


    # Step 4: Form the matrix Sigma
    Sigma = np.zeros(A.shape)
    for i in range(len(Sigma)):
        for j in range(len(Sigma[0])):
            if (i == j) and (eigenvalue[i] != 0):
                Sigma[i,j] = sqrt(eigenvalue[i])
    # print('Sigma: \n',Sigma)

    # Step 5: Form the matrix U
    u = np.zeros((A.shape[0],A.shape[0]))
    for i in range(Sigma.shape[0]):
        u[i] = np.matmul(np.multiply(1/(sqrt(eigenvalue[i])),A), v[i])
    # print('u: \n', u)
    U = u.transpose()
    # print("U = \n", U)
    return U, Sigma, Vt

### DRIVER ###
# # Compute SVD
# img_name = 'momo_kecil_gray.png'
# path = 'in/' + img_name
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # print(img)
# print(gray)
# print(img.shape)
# print(gray.shape)
start = datetime.datetime.now()
A = np.array([[33,222,111],[222,111,44]])
print(svd(A))
end = datetime.datetime.now()
print((end-start))
# Percobaan 1: 4783


# print(svd(gray))
# Step 6
# # TESTING BENER APA ENGGA
# print("---TESTING---")
# print("A: \n",A)
# A_test = np.matmul(np.matmul(U,Sigma), Vt)
# print("A_test: \n",A_test)
# print("U: \n",U)
# print("Sigma: \n",Sigma)
# print("Vt: \n",Vt)

# # BANDINGIN SAMA HASIL LIBRARY
# U_lib, S_lib, Vt_lib = np.linalg.svd(A)
# print("U_lib: \n",U_lib)
# print("S_lib: \n",S_lib)
# print("Vt_lib: \n",Vt_lib)