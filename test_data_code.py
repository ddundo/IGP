import numpy as np
from obspy.imaging.beachball import beachball
import matplotlib.pyplot as plt

data = np.genfromtxt('TEST_DATA.txt')

mts = np.array(data[:, 3:9])
exp = np.array(data[:, 9])
mts = [mts[i, :] * 10**exp[i] for i in range(len(exp))]
mt_sum = np.sum(mts, axis=0)

print(sum)
beachball(sum)
plt.show()

### Mrr -9.2557e26

# my results: [-9.255760e+26  1.911197e+27 -9.823480e+26 -1.706900e+25  3.282760e+26
#  -1.650034e+27]