import numpy as np
from scipy.spatial import distance


def mul(A: list, num: float) -> list:
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j] * num
    return A


def mat_sum(A: list, B: list) -> list:
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j] + B[i][j]
    return A

def dist_new(f1, f2):
    return np.linalg.norm(f1, f2)

dist_M_temp= [
    [1, 2, 3],
    [2, 9, 3],
    [3, 2, 3]
]

dist_M= [
    [1, 1, 1],
    [2, 3, 3],
    [3, 2, 3]
]
diff = 0.2
f1 = [1, 2, 3]
f2 = [1, 2, 2]
print(dist_new(f1, f2))