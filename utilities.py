import numpy as np
from math import sin, cos, asin, sqrt, log, radians
from uncertainties import unumpy, umath


# def haversine(pos1, pos2):
#     lon1, lat1, lon2, lat2 = map(radians, [pos1[0], pos1[1], pos2[0], pos2[1]])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     km = 6371 * c
#     return km


def _radius(theta):
    r0 = 6356.752  # WGS84
    r1 = 6378.137
    return sqrt(((r1**2 * cos(theta))**2 + (r0**2 * sin(theta))**2) /
                ((r1 * cos(theta))**2 + (r0 * sin(theta))**2))


def _spherical2cart(pos):
    r, theta, phi = pos
    theta = radians(theta)
    phi = radians(phi)
    return (
        r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta)
    )


def distance(pos1, pos2):
    x1, y1, z1 = _spherical2cart(pos1)
    x2, y2, z2 = _spherical2cart(pos2)
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 - (z2 - z1) ** 2)


def dynecm2nm(x):
    return x * 10E-7


def nm2dynecm(x):
    return x * 10E7


def sum_m0(tensors):
    return np.sum([tensor.m0 for tensor in tensors])


def tensor_sum(tensors):
    mt_sum = _sum_mt(tensors)
    try:
        exp = log(np.max(mt_sum), 10) // 1
    except TypeError:
        exp = log(np.max(unumpy.nominal_values(mt_sum)), 10) // 1
    mt_sum /= 10**exp
    tensor = MomentTensor(mt_sum, exp)
    return tensor


def row2mt(data):
    date = data[0]
    pos = (_radius(data[1]), data[1], data[2])  # r, lat, lon
    depth = data[3]
    exp = data[8]
    mt = np.array([[data[9], data[15], data[17]],
                   [data[15], data[11], data[19]],
                   [data[17], data[19], data[13]]])
    mt_err = np.array([[data[10], data[16], data[18]],
                       [data[16], data[12], data[20]],
                       [data[18], data[20], data[14]]])

    return MomentTensor(mt, exp, mt_err, pos, depth, date)


def strain_tensor(tensors, volume, mu=3.3e10):
    return _sum_mt(tensors) / (2 * mu * volume)  # TODO


def plate_vel(tensors, l, w, t, mu=3.3e10):
    return sum_m0(tensors) / (mu * l * w * t)


def _sum_mt(tensors):
    return np.sum([tensor.mt * 10 ** tensor.exp for tensor in tensors], axis=0)


def _calculate_eigh(mt):
    return np.linalg.eigh(np.array(mt, dtype=np.float_))


def _calculate_m0(mt):
    t, b, p = sorted(_calculate_eigh(mt.mt * 10 ** mt.exp)[0])
    return sqrt((t ** 2 + b ** 2 + p ** 2) / 2)


def _calculate_mw(mt):
    return 2 / 3 * log(mt.m0, 10) - 10.7


class MomentTensor(object):
    def __init__(self, mt, exp, mt_err=None, pos=None, depth=None, date=None):
        self.exp = exp
        self.pos = pos
        self.depth = depth
        self.date = date

        if np.shape(mt) == (3, 3):
            self.mt = mt
        else:
            self.mt = np.array([[mt[0], mt[3], mt[4]],
                                [mt[3], mt[1], mt[5]],
                                [mt[4], mt[5], mt[2]]])

        if mt_err is not None:
            if np.shape(mt_err) == (3, 3):
                self.mt_err = mt_err
            else:
                self.mt_err = np.array([[mt_err[0], mt_err[3], mt_err[4]],
                                        [mt_err[3], mt_err[1], mt_err[5]],
                                        [mt_err[4], mt_err[5], mt_err[2]]])

    @property
    def m0(self):
        return _calculate_m0(self)

    @property
    def mw(self):
        return _calculate_mw(self)

    @property
    def r(self):
        return self.pos[0]

    @property
    def lat(self):
        return self.pos[1]

    @property
    def lon(self):
        return self.pos[2]
