import numpy as np
import random as rdm

P3d = np.zeros((10,10), dtype=float)

for i in range(0, 10):
    for j in range(0, 10):
        if (j>=i-3 and j<=i+3):
            P3d[i][j] = rdm.randint(1, 9)
print(P3d)

def createTransit(matrix, bandWidth):
    ab = []
    for i in range(bandWidth, -bandWidth-1, -1):
        ac = []
        if i>=0:
            for j in range(0, i):
                ac.append(0)
            for j in range(0, len(np.diagonal(matrix, i))):
                ac.append(np.diagonal(matrix, i)[j])
        else:
            for j in range(0, len(np.diagonal(matrix, i))):
                ac.append(np.diagonal(matrix, i)[j])
            for j in range(i, 0):
                ac.append(0)
        ab.append(ac)
    ab = np.array(ab)
    # print(ab)
    return ab

createTransit(P3d, 3)