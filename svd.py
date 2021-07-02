from math import sqrt
import numpy as np

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

    return U, Sigma, Vt