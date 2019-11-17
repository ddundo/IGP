from uncertainties import unumpy
import numpy as np
import utilities
import pandas as pd

df = pd.read_pickle('gcmt_lat5075_lon130180.pkl')
mat = df.to_numpy()

mts_6 = np.transpose(np.array([mat[:, 9], mat[:, 11], mat[:, 13], mat[:, 15], mat[:, 17], mat[:, 19]]))
mts_6_unc = np.transpose(np.array([mat[:, 10], mat[:, 12], mat[:, 14], mat[:, 16], mat[:, 18], mat[:, 20]]))
iexp = mat[:, 8]

mts_unc = [unumpy.uarray(mts_6[i], mts_6_unc[i]) for i in range(len(mts_6))]

mts = [utilities.MomentTensor(mt, exp) for mt, exp in zip(mts_unc, iexp)]

# print(mts[0].mt + mts[1].mt)
# print(np.sum([mts[0].mt, mts[1].mt], axis=0))

sum = utilities.tensor_sum(mts)
print(sum.mt)
print(sum.exp)