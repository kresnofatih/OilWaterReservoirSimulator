import numpy as np
import random as rdm

P3d = np.zeros((10,10), dtype=float)

for i in range(0, 10):
    for j in range(0, 10):
        if (j>=i-3 and j<=i+3):
            P3d[i][j] = rdm.randint(1, 9)
print(P3d)

# print(np.diagonal(P3d, 3))
# print(np.diagonal(P3d, 2))
# print(np.diagonal(P3d, 1))
# print(np.diagonal(P3d, 0))
# print(np.diagonal(P3d, -1))
# print(np.diagonal(P3d, -2))
# print(np.diagonal(P3d, -3))

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
    print(ab)
    return

createTransit(P3d, 3)

# a = np.ones((3,3,3), dtype=float)
# b = np.zeros((3,3,3), dtype=float)

# b+=100

# a += 10
# b[2][2][2] = 300

# print(abs(a-b))
# print(np.amax(abs(a-b)))

# print(min([2,3]))