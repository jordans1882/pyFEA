import splipy
import xalglib
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

randDistX = np.random.rand(1000)
randDistY = np.random.rand(1000)


#k = 10
#print(np.array([0.0]))
#knot_vec = np.concatenate((np.array([0.0, 0.0]), np.linspace(0.0, 1.0, k), np.array([1.0, 1.0])))
#tseq = np.linspace(0.0, 1.0, 50)
#print(randDist)
bsplineSequence = splipy.BSplineBasis(3, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1], -1)
x = bsplineSequence.evaluate(randDistX, 0, True, True)
#print(x[:,:])
#print(bsplineSequence[4])
#print(len(bsplineSequence))
#plt.plot(tseq, x[:,1])
#for i in range(len(x[1,:])):
#    plt.plot(tseq, x[:, i])
#plt.show()
"""print(x._shape_as_2d)
x = x.reshape(x.shape[0], 1)"""


transposeX = np.transpose(x)
print(transposeX.shape)
firstPart = (transposeX)@x
negated = np.linalg.pinv(firstPart)
allXs = negated@(transposeX)
complete = allXs@(randDistY)
print(complete)