import obspy.imaging.beachball as bb
import numpy as np
import pandas as pd

# test_data = np.genfromtxt('./TEST_DATA.txt')
# moment_tensors = [bb.MomentTensor(data[i, 3:9], data[i, -4]) for i in range(len(data))]


# lon lat depth mrr mtt mpp mrt mrp mtp iexp 'X' 'Y' name
data = np.genfromtxt('./gcmt_5075_180130.txt')
# The relation to Aki and Richards x,y,z equals North,East,Down convention is as follows:
# Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy
# print(data[1, 3:8], data[1, -4])

moment_tensors = [bb.MomentTensor(data[i, 3:9], data[i, -4]) for i in range(len(data))]

print(moment_tensors[1].mt)

### TESTING ###

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

mt_test = bb.MomentTensor([4.810, -4.523, -0.287,   4.885,   3.476,  -2.651], 24)

# eigval, eigvct = np.linalg.eig(mt_test.mt)
# print(eigval) works
# print(eigvct) works

print(bb.mt2axes(mt_test)[0].dip)# bb.mt2axes(mt_test).strike, bb.mt2axes(mt_test).val)
