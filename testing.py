import numpy as np
from utilities import MomentTensor, princax
import obspy.imaging.beachball as bb
from math import sqrt, log

# TESTING

# Event name: 110477B
# Region name: ANDREANOF IS., ALEUTIANS
# Date (y/m/d): 1977/11/4

# Information on data used in inversion

# Wave    nsta  nrec  cutoff
# Body      10    27   45
# Mantle     0     0   0
# Surface    0     0   0
# Timing and location information

#          hr  min   sec       lat     lon    depth   mb   Ms
# MLI      18    7  31.30     51.43  -175.56   33.0  5.4  5.4
# CMT      18    7  34.20     51.47  -175.41   32.2
# Error              0.30      0.03     0.05    2.0
# Assumed half duration:  3.3

# Mechanism information
# Exponent for moment tensor:  24    units: dyne-cm
#          Mrr     Mtt     Mpp     Mrt     Mrp     Mtp
# CMT     4.810  -4.523  -0.287   4.885   3.476  -2.651
# Error   0.133   0.201   0.132   0.373   0.279   0.158

# Mw = 5.9   Scalar Moment = 8.01e+24
# Fault plane:  strike=255    dip=22   slip=107
# Fault plane:  strike=56    dip=69   slip=83
# Eigenvector:  eigenvalue:  7.58   plunge: 65   azimuth: 315
# Eigenvector:  eigenvalue:  0.88   plunge:  6   azimuth:  59
# Eigenvector:  eigenvalue: -8.45   plunge: 24   azimuth: 152

mt = MomentTensor([4.810, -4.523, -0.287,   4.885,   3.476,  -2.651], 24)

eigvals, eigvcts = np.linalg.eigh(np.array(mt.mt * 10**mt.exp, dtype=float))
print(eigvals)   #[-8.45266194  7.57685043  0.87581151]
print(eigvcts)   #[[-0.40938256  0.90576372 -0.1095354 ]
                 # [ 0.80215416  0.30012763 -0.51620937]
                 # [ 0.43468912  0.29919139  0.84942915]] in columns

t, b, p = princax(mt)
print(t, b, p)

print(bb.mt2axes(mt)[0].__dict__)  # {'val': 7.58, 'strike': 315, 'dip': 64.9}
print(bb.mt2axes(mt)[1].__dict__)  # {'val': 0.88, 'strike': 58.7, 'dip': 6.29}
print(bb.mt2axes(mt)[2].__dict__)  # {'val': -8.45, 'strike': 151.55, 'dip': 24.2}

print(bb.mt2plane(mt).__dict__)    # {'strike': 56.34, 'dip': 69.45, 'rake': 83.28}
print(bb.aux_plane(56.344996989653225, 69.45184548172541, 83.28228402625453))
#(254.8956832264013, 21.573156870706903, 107.33165895106156) strike, dip, slip aux plane


M0 = sqrt(np.sum(eigvals ** 2) / 2)
print('M0: ', M0)  # 8.050565259657235 * 10 ** 24 Scalar Moment

M_W = 2 / 3 * log(M0, 10) - 10.7
print(M_W)  # 5.903884249895878

T, B, P = eigvals
f_CLVD = abs(B) / max(abs(T), abs(P))
print(f_CLVD)  # 0.10361369212705232

print(mt.fclvd)
