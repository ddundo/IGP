import pandas as pd
import numpy as np
import utilities
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

df = pd.read_pickle("./gcmt_all_earthquakes.pkl")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# print(df)

mat = df.to_numpy()
global_mts_6 = np.transpose(np.array([mat[:, 10], mat[:, 12], mat[:, 14], mat[:, 16], mat[:, 18], mat[:, 20]]))
iexp = mat[:, 9]

global_MomentTensors = [utilities.MomentTensor(global_mts_6[i, :], iexp[i]) for i in range(len(global_mts_6))]

eigvals = [sorted(np.linalg.eigh(MT.mt)[0]) for MT in global_MomentTensors]  # T, B, P
eigvcts = [np.linalg.eigh(MT.mt)[1] for MT in global_MomentTensors]
# princ_moments = np.array([sorted(eigvals[i]) for i in range(len(eigvals))])  # T, B, P
f_CLVD = [abs(PM[1]) / max(abs(PM[0]), abs(PM[2])) for PM in eigvals]

print('Global f_CLVD mean: ', np.mean(f_CLVD, axis=0))

# Alaska
df = pd.read_pickle('gcmt_lat5075_lon130180.pkl')
mat = df.to_numpy()
mts_6 = np.transpose(np.array([mat[:, 10], mat[:, 12], mat[:, 14], mat[:, 16], mat[:, 18], mat[:, 20]]))
iexp = mat[:, 9]
iexp = np.reshape(np.repeat(iexp, 6), np.shape(mts_6))
mts_6 = mts_6 * iexp
global_MomentTensors = [utilities.MomentTensor(global_mts_6[i, :], iexp[i]) for i in range(len(mts_6))]
eigvals = [sorted(np.linalg.eigh(MT.mt)[0]) for MT in global_MomentTensors]  # T, B, P
eigvcts = [np.linalg.eigh(MT.mt)[1] for MT in global_MomentTensors]
# princ_moments = np.array([sorted(eigvals[i]) for i in range(len(eigvals))])  # T, B, P
f_CLVD_al = [abs(PM[1]) / max(abs(PM[0]), abs(PM[2])) for PM in eigvals]
print('Alaska f_CLVD mean: ', np.mean(f_CLVD_al, axis=0))
###


plt.hist(f_CLVD, 50, alpha=0.5, label='Global', density=True)
plt.hist(f_CLVD_al, 50, alpha=0.5, label='Alaska', density=True)
plt.legend(loc='upper right')
plt.xlim(0, 0.5)
plt.xlabel('\\textit{f_{CLVD}}', fontsize=16)
plt.show()
