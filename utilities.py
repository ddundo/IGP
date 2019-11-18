import numpy as np
from numpy import cos, sin, tan, arctan, radians, degrees, arcsin, arctan2, sqrt, arccos
from uncertainties import unumpy
from math import log


def haversine(pos1, pos2):
    lon1, lat1, lon2, lat2 = map(radians, [pos1[0], pos1[1], pos2[0], pos2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * arcsin(sqrt(a))
    km = 6371 * c
    return km


def vincenty(lat1, lon1, lat2, lon2):
    a = 6378137.0  # equatorial radius in meters
    f = 1 / 298.257223563  # ellipsoid flattening
    b = (1 - f) * a
    tolerance = 1e-11  # to stop iteration

    phi1, phi2 = lat1, lat2
    U1 = arctan((1 - f) * tan(phi1))
    U2 = arctan((1 - f) * tan(phi2))
    L1, L2 = lon1, lon2
    L = L2 - L1

    lambda_old = L + 0

    while True:
        t = (cos(U2) * sin(lambda_old)) ** 2
        t += (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(lambda_old)) ** 2
        sin_sigma = t ** 0.5
        cos_sigma = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(lambda_old)
        sigma = arctan2(sin_sigma, cos_sigma)

        sin_alpha = cos(U1) * cos(U2) * sin(lambda_old) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha ** 2
        cos_2sigma_m = cos_sigma - 2 * sin(U1) * sin(U2) / cos_sq_alpha
        C = f * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha)) / 16

        t = sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2))
        lambda_new = L + (1 - C) * f * sin_alpha * t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha * ((a ** 2 - b ** 2) / b ** 2)
    A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    t = cos_2sigma_m + 0.25 * B * (cos_sigma * (-1 + 2 * cos_2sigma_m ** 2))
    t -= (B / 6) * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2)
    delta_sigma = B * sin_sigma * t
    s = b * A * (sigma - delta_sigma)

    return s


def princax(tensor):
    mt = np.array(tensor.mt * 10 ** tensor.exp, dtype=np.float_)
    val, vct = np.linalg.eigh(mt)
    pl = arcsin(-vct[0])
    az = arctan2(vct[2], -vct[1])
    for i in range(3):
        if pl[i] <= 0:
            pl[i] = -pl[i]
            az[i] += np.pi
        if az[i] < 0:
            az[i] += 2 * np.pi
        if az[i] > 2 * np.pi:
            az[i] -= 2 * np.pi

    pl = degrees(pl)
    az = degrees(az)

    t = (val[0], pl[0], az[0])
    b = (val[1], pl[1], az[1])
    p = (val[2], pl[2], az[2])
    return t, b, p


def kaverina(dipt, dipb, dipp):
    """x and y for the Kaverina diagram"""
    zt = sin(radians(dipt))
    zb = sin(radians(dipb))
    zp = sin(radians(dipp))
    l = 2 * sin(0.5 * arccos((zt + zb + zp) / sqrt(3)))
    n = sqrt(2 * ((zb - zp) ** 2 + (zb - zt) ** 2 + (zt - zp) ** 2))
    x = sqrt(3) * (l / n) * (zt - zp)
    y = (l / n) * (2 * zb - zp - zt)
    return x, y


def radius(theta):
    r0 = 6356.752  # WGS84
    r1 = 6378.137
    return sqrt(((r1 ** 2 * cos(theta)) ** 2 + (r0 ** 2 * sin(theta)) ** 2) /
                ((r1 * cos(theta)) ** 2 + (r0 * sin(theta)) ** 2))


def spherical2cart(pos):
    r, theta, phi = pos
    theta = radians(theta)
    phi = radians(phi)
    return (
        r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta)
    )


def distance(pos1, pos2):
    x1, y1, z1 = spherical2cart(pos1)
    x2, y2, z2 = spherical2cart(pos2)
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


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
    mt_sum /= 10 ** exp
    tensor = MomentTensor(mt_sum, exp)
    return tensor


def row2mt(data):
    date = data[0]
    pos = (radius(data[1]), data[1], data[2])  # r, lat, lon
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


def _m0(mt):
    eigvals, _ = np.linalg.eigh(mt.mt)
    eigvals = eigvals * 10 ** mt.exp
    return sqrt(np.sum(eigvals ** 2) / 2)


def _mw(mt):
    return 2 / 3 * log(mt.m0, 10) - 10.7


class MomentTensor(object):
    def __init__(self, mt, exp, mt_err=None, pos=None, depth=None, date=None):
        self.exp = exp
        self.pos = pos
        self.depth = depth
        self.date = date

        if np.shape(mt) == (3, 3):
            self.mt = np.array(mt, dtype=np.float_)
        else:
            self.mt = np.array([[mt[0], mt[3], mt[4]],
                                [mt[3], mt[1], mt[5]],
                                [mt[4], mt[5], mt[2]]], dtype=np.float_)

        if mt_err is not None:
            if np.shape(mt_err) == (3, 3):
                self.mt_err = np.array(mt_err, dtype=np.float_)
            else:
                self.mt_err = np.array([[mt_err[0], mt_err[3], mt_err[4]],
                                        [mt_err[3], mt_err[1], mt_err[5]],
                                        [mt_err[4], mt_err[5], mt_err[2]]], dtype=np.float_)

    @property
    def m0(self):
        return _m0(self)

    @property
    def mw(self):
        return _mw(self)

    @property
    def r(self):
        return self.pos[0]

    @property
    def lat(self):
        return self.pos[1]

    @property
    def lon(self):
        return self.pos[2]

    @property
    def axes(self):
        return np.array(princax(self))

    @property
    def fclvd(self):
        t, b, p = self.axes[:, 0]
        return abs(b) / max(abs(t), abs(p))
