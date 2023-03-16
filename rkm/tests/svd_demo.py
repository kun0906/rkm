

import numpy as np
A = np.asarray([[1, 2], [3, 4], [5,6]])

U, S, V = np.linalg.svd(A,full_matrices=True)
print(U, S**2, V)
# U2, S2, V2 = np.linalg.svd(A@A.T,full_matrices=True)
# print(U2, S2, V2)
# U3, S3, V3 = np.linalg.svd(A.T@A,full_matrices=True)
# print(U3, S3, V3)
W1, V1 = np.linalg.eig(A@A.T)
print(W1, V1)

W2, V2 = np.linalg.eig(A.T@A)
print(W2, V2)

