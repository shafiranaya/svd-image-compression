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
    # Atranspose_A = np.array(Atranspose_A,dtype=np.float64)
    # print(Atranspose_A)
    # A_Atranspose = np.matmul(A,A.transpose())

    # Step 2: Determine eigenvalues of A'A
    # eigenvalue = np.linalg.eigvalsh(Atranspose_A)
    # eig_val = []
    # eigenvects = sp.Matrix(Atranspose_A).eigenvects()
    # for i in range(len(eigenvects)):
    #     eig_val.append(eigenvects[i][0])
    # eig_val = sorted(eig_val,reverse=True)
    # print("eig_val\n",eig_val)

    eigenvalue  = np.linalg.eig(Atranspose_A)[0].real
    # for i in range(len(eigenvalue)):
    #     if (eigenvalue[i]) < 0:
    #         eigenvalue[i] *= -1
    
    # for i in range(len(eigenvalue)):
    #     if (eigenvalue[i] < 0):
    #         eigenvalue[i] = 0
    #     else:
    #         eigenvalue[i] = round(eigenvalue[i])

    # for val in eigenvalue:
    #     val = abs(val)
    eigenvalue = sorted(eigenvalue,reverse=True)
    print("Eigenvalue=\n",eigenvalue)
    print("Len eigenvalue =", len(eigenvalue))

    print(Atranspose_A.shape)
    print("Determinant: ", np.linalg.det(Atranspose_A))

    # eigenvalue_lain = []
    
    # eigenvects = sp.Matrix(Atranspose_A).eigenvects()
    # eigenvects = sorted(eigenvects,reverse=True) # decreasing magnitude
    # # print("eigenvector:\n",eigenvects)
    # for i in range(len(eigenvects)):
    #     eigenvalue_lain.append(eigenvects[i][0])
    # matriks = sp.Matrix(Atranspose_A)
    # eigenvalue_lain = matriks.eigenvals()
    # eigenvalue_lain = scipy.sparse.linalg.eigs(Atranspose_A)

    # eigenvalue_lain = scipy.linalg.eigh(Atranspose_A)[0].real
    # # eigenvalue_lain = sorted(),reverse=True)
    # print("Eigenvalue lain=\n",eigenvalue_lain)
    # print("Len eigenvalue lain=", len(eigenvalue_lain))
    # count_pos = 0
    # for val in eigenvalue_lain:
    #     if val > 0:
    #         count_pos += 1
    #     else:
    #         val = abs(val)
    # print("Positive eigenvalue=\n",count_pos)
    # print("Eigenvalue lain=\n",eigenvalue_lain)
    # eigenvalue_lain = sorted(eigenvalue_lain,reverse=True)

    # for val in eigenvalue:
    #     if val < 0:

    # eigenvalue = []
    # print(eig_val)
    # eigenvects = sp.Matrix(Atranspose_A).eigenvects()
    # eigenvects = sorted(eigenvects,reverse=True) # decreasing magnitude
    # for i in range(len(eigenvects)):
    #     eigenvalue.append(eigenvects[i][0])
    # eigenvects = sorted(sp.Matrix(Atranspose_A).eigenvects(),reverse=True) # decreasing magnitude
    # for i in range(len(eigenvects)):
    #     eigenvalue.append(eigenvects[i][0])
    # print('Eigenvalue: ',eigenvalue)

    # Step 3: Form the matrix V'
    eigenvector = np.linalg.eigh(Atranspose_A)[1]
    eigenvector = eigenvector.transpose()
    # Vt = np.flip(eigenvector,0)
    # # print("Vt = \n", Vt)
    # v = np.zeros((Vt.shape))
    # for i in range(len(Vt)):
    #     v[i] = Vt[i]
    Vt = np.flip(eigenvector,0)
    # print("Vt = \n", Vt)
    v = np.zeros((Vt.shape))
    for i in range(len(Vt)):
        v[i] = Vt[i]

    # Step 4: Form the matrix Sigma
    Sigma = np.zeros(A.shape)
    for i in range(len(Sigma)):
        for j in range(len(Sigma[0])):
            if (i == j) and (eigenvalue[i] > 0):
                Sigma[i,j] = sqrt((eigenvalue[i]))
                # Sigma[i,j] = sqrt((eig_val[i]))
    # print('Sigma: \n',Sigma)

    # Step 5: Form the matrix U
    u = np.zeros((A.shape[0],A.shape[0]))
    for i in range(Sigma.shape[0]):
        if (eigenvalue[i] > 0):
            u[i] = np.matmul(np.multiply(1/(sqrt((eigenvalue[i]))),A), v[i])
        # u[i] = np.matmul(np.multiply(1/(sqrt((eig_val[i]))),A), v[i])
    # print('u: \n', u)
    U = u.transpose()
    # print("U = \n", U)
    # print("---TESTING---")
    # print("A: \n",A)
    # U, Sigma, Vt = svd(A)
    # A_test = np.matmul(np.matmul(U,Sigma), Vt)
    # print("A_test: \n",A_test)
    print("U: \n",U)
    print("Sigma: \n",Sigma)
    print("Vt: \n",Vt)

    # BANDINGIN SAMA HASIL LIBRARY
    U_lib, S_lib, Vt_lib = np.linalg.svd(A)
    print("U_lib: \n",U_lib)
    S_lib = sorted(S_lib, reverse=True)
    print("S_lib: \n",S_lib)
    print("Vt_lib: \n",Vt_lib)
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