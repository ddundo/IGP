import numpy as np
from obspy.imaging.beachball import beachball
from numpy.linalg import eig
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# lon lat depth mrr mtt mpp mrt mrp mtp iexp 'X' 'Y' name
data = np.genfromtxt('gcmt_5075_180130.txt')

print(data)

print(np.shape(data))

moment_tensors = [np.reshape([data[i, 3], data[i, 6], data[i, 7], data[i, 6], data[i, 4], data[i, 8],
                              data[i, 7], data[i, 8], data[i, 5]], (3, 3)) for i in range(np.shape(data)[0])]

print('moment tensors: ', np.shape(moment_tensors), len(moment_tensors))
print(moment_tensors[0])

eigvals = [eig(moment_tensors[i])[0] for i in range(len(moment_tensors))]
eigvcts = [eig(moment_tensors[i])[1] for i in range(len(moment_tensors))]

print(np.shape(eigvals), np.shape(eigvcts))

princ_moments = [sorted(eigvals[i]) for i in range(len(eigvals))]  # T, B, P

print(np.shape(princ_moments))

f_CLVD = [abs(princ_moments[i, 1]) / max(abs(princ_moments[i, 0]),
                                         abs(princ_moments[i, 2])) for i in range(len(princ_moments))]
print('f_CLVD: ', np.shape(f_CLVD), np.mean(f_CLVD))
# print(f_CLVD)

sum = [np.sum(data[:, 3], axis=0), np.sum(data[:, 4], axis=0), np.sum(data[:, 5], axis=0), np.sum(
    data[:, 6], axis=0), np.sum(data[:, 7], axis=0), np.sum(data[:, 8], axis=0)]

print(sum)

print(np.max(data[:, 2]))

mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]

test = [-2.39, 0.57, -2.94, 0.57, 1.04, -0.94, -2.94, -0.94, 1.35]

test = np.reshape(test, (3, 3))

print(test)

w, v = eig(test)
print('w: ', w)
print('v: ', v)
beachball(mt)

print('done')


mt = [1.73, -0.427, -0.61, 2.98, -2.4, 0.426]
mt_test = [1.73, 2.98, -2.4, 2.98, -0.427, 0.426, -2.4, 0.426, -0.61]
mt_test = np.reshape(mt_test, (3, 3))
print(mt_test)
w, v = eig(mt_test)
print('w: ', w)
print('v: ', v)
beachball(mt)

p0 = [0.81400949,  0.57927701, -0.04273997]
p1 = [0.46635507, - 0.60791683,  0.64261192]
p2 = [-0.34626796,  0.5430242,   0.76499884]

origin = [0, 0, 0]
X, Y, Z = zip(origin, origin, origin)
U, V, W = zip(p0, p1, p2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.01)
plt.show()
