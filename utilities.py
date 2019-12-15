import numpy as np
from numpy import cos, sin, tan, arctan, radians, degrees, arcsin, arctan2, sqrt, arccos
from uncertainties import unumpy
from math import log
from scipy.linalg import lstsq
from datetime import datetime
import matplotlib.pyplot as plt
from PyGEL3D import gel


def dynecm2nm(x):
    return x * 1e-7


def nm2dynecm(x):
    return x * 1e7


def spherical2cart(pos):
    r, theta, phi = pos
    theta = radians(theta)
    phi = radians(phi)
    return (
        r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta)
    )


def cart2spherical(pos):
    x, y, z = pos
    r = sqrt(np.sum(np.array(pos) ** 2))
    theta = degrees(arctan2(sqrt(x**2 + y**2), z))
    phi = degrees(arctan2(y, x))

    return r, theta, phi


def distance(pos1, pos2):
    x1, y1, z1 = spherical2cart(pos1)
    x2, y2, z2 = spherical2cart(pos2)
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def haversine(pos1, pos2):
    lon1, lat1, lon2, lat2 = map(radians, [pos1[0], pos1[1], pos2[0], pos2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * arcsin(sqrt(a))
    km = 6371 * c
    return km


def vincenty(lat1, lon1, lat2, lon2):
    """
    Written by https://www.johndcook.com/
    """
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


def distance_from_hull(hull, mts):
    points = [spherical2cart(mt.pos) for mt in mts]
    m = gel.Manifold()
    for s in hull.simplices:
        m.add_face(hull.points[s])

    dist = gel.MeshDistance(m)
    res = []

    for p in points:
        d = dist.signed_distance(p)
        if dist.ray_inside_test(p):
            if d > 0:
                d *= -1
        else:
            if d < 0:
                d *= -1
        res.append(d)
    res = np.round(np.array(res) / 1000, 2)  # m2km

    return res


def rotate(data, strike):
    data = data.copy()
    theta = 270 - strike if strike < 270 else 90 - strike
    theta = radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

    for i in range(len(data)):
        data[i, :2] = Rz.dot(data[i, :2])

    return data


def planefit(tensors):
    data = np.row_stack([spherical2cart(tensor.pos) for tensor in tensors])

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    dx = 1e4
    res = abs(int((mn[0] - mx[1]) / dx))  # 20
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], res), np.linspace(mn[1], mx[1], res))
    XX = X.flatten()
    YY = Y.flatten()

    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    C, res, _, _ = lstsq(A, data[:, 2])
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

    # print('COEFFICIENTS: ', C)

    from scipy import integrate

    b = C[1]; c = C[2]; d = C[3]; e = C[4]; f = C[5]
    fun = lambda x, y: sqrt((b + d * y + 2 * e * x) ** 2 + (c + d * x + 2 * f * y) ** 2 + 1)
    # print('AREA: ')
    area = integrate.dblquad(fun, mn[0], mx[0], lambda x: mn[1], lambda x: mx[1])

    return X, Y, Z, area


# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
    return x / magnitude, y / magnitude, z / magnitude


# area of polygon poly
def poly_area(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i + 1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


def surface_area(tensors=None, x=None, y=None, z=None):
    if tensors is not None:
        x, y, z = planefit(tensors)
    area = 0

    for j in range(len(y) - 1):
        for i in range(len(x) - 1):
            dA = poly_area([[x[i, j], y[i, j], z[i, j]],
                            [x[i + 1, j], y[i + 1, j], z[i + 1, j]],
                            [x[i, j + 1], y[i, j + 1], z[i, j + 1]],
                            [x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]]])
            area += dA

    return area


def avg_strain_tensor(tensors, area=None):
    if tensors is not None:
        area = surface_area(tensors)
    volume = area * 1e5
    mu = 3.3e10
    tensor = tensor_sum(tensors)
    strain_tensor = tensor.mt_e / (2 * mu * volume)

    return strain_tensor


def second_invariant(eps):
    eigval = np.linalg.eigh(eps)[0]
    I1 = np.sum(eigval)
    I2 = eigval[2] * eigval[1] + eigval[2] * eigval[0] + eigval[1] * eigval[0]
    J2 = I1 ** 2 - 2 * I2

    return J2


def plate_velocity(tensors):
    dates = [mt.date for mt in tensors]
    mu = 3.3e10
    t = (max(dates) - min(dates)).total_seconds() / (3600 * 24 * 365)
    x, y, z = planefit(tensors)
    l = max(abs(max(x.flatten()) - min(x.flatten())), abs(max(y.flatten()) - min(y.flatten())))
    w = abs(max(z.flatten()) - min(z.flatten()))
    print(l)
    print(w)
    v = sum_m0(tensors) / (mu * l * w * t)

    return v


def plate_velocity_usgs(usgs_df, hull):
    import pandas as pd

    mu = 3.3e10
    times = usgs_df[['time']].to_numpy().flatten()
    t = pd.Timedelta(max(times) - min(times)).value / 3.154e+16  # convert ns to year


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def princax(tensor):
    mt = np.array(tensor.mt_e, dtype=np.float_)
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


def angle(v1, v2, acute):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if acute == True:
        return angle
    else:
        return 2 * np.pi - angle


def kaverina(dipt, dipb, dipp):
    zt = sin(radians(dipt))
    zb = sin(radians(dipb))
    zp = sin(radians(dipp))
    l = 2 * sin(0.5 * arccos((zt + zb + zp) / sqrt(3)))
    n = sqrt(2 * ((zb - zp) ** 2 + (zb - zt) ** 2 + (zt - zp) ** 2))
    x = sqrt(3) * (l / n) * (zt - zp)
    y = (l / n) * (2 * zb - zp - zt)
    return x, y


def kaverina_base():
    fig = plt.figure()
    plt.axes().set_aspect('equal')

    deg = 90 - 40
    degg = sin(radians(90 - deg)) * cos(radians(deg))
    B = degrees(arcsin(sqrt(np.linspace(0, 1, 51) * degg)))
    P = degrees(arcsin(sqrt((1 - np.linspace(0, 1, 51)) * degg)))
    X, Y = kaverina(deg, B, P)
    plt.plot(X, Y, '--', color='grey', linewidth=1)

    deg = 90 - 30
    degg = sin(radians(90 - deg)) * cos(radians(deg))
    B = degrees(arcsin(sqrt(np.linspace(0, 1, 51) * degg)))
    P = degrees(arcsin(sqrt((1 - np.linspace(0, 1, 51)) * degg)))
    X, Y = kaverina(P, B, deg)
    plt.plot(X, Y, '--', color='grey', linewidth=1)
    X, Y = kaverina(P, deg, B)
    plt.plot(X, Y, '--', color='grey', linewidth=1)

    tickx, ticky = kaverina(range(0, 91, 10), np.zeros((1, 10)), range(90, -1, -10))
    # plt.plot(X[0], Y[0], color='black', linewidth=2)
    plt.scatter(tickx[0][1:], ticky[0][1:], marker=3, c='black', linewidth=1)
    for i in range(1, 10):
        plt.text(tickx[0][i] - 0.04, ticky[0][i] - 0.04, i * 10, fontsize=9, verticalalignment='top')
    plt.text(0, -0.75, 'T axis plunge', fontsize=9, horizontalalignment='center')
    T = degrees(arcsin(sqrt(np.linspace(0, 1, 101))))
    P = degrees(arcsin(sqrt(1 - (np.linspace(0, 1, 101)))))

    X, Y = kaverina(T, 0., P)
    plt.plot(X, Y, color='black', linewidth=1)
    X, Y = kaverina(P, T, 0.)
    plt.plot(X, Y, color='black', linewidth=1)
    X, Y = kaverina(0., P, T)
    plt.plot(X, Y, color='black', linewidth=1)

    tickx, ticky = kaverina(np.zeros((1, 10)), range(0, 91, 10), range(90, -1, -10))
    # plt.plot(X[0], Y[0], color='black', linewidth=2)
    plt.scatter(tickx[0][1:], ticky[0][1:], marker=0, c='black', linewidth=1)
    for i in range(1, 10):
        plt.text(tickx[0][i] - 0.04, ticky[0][i] - 0.02, i * 10, fontsize=9, horizontalalignment='right')
    plt.text(-0.63, 0.25, 'B axis plunge', fontsize=9, horizontalalignment='center', rotation=60)

    plt.axis('off')

    return fig


def triangle(tdip, bdip, pdip):
    tdip = radians(tdip)
    bdip = radians(bdip)
    pdip = radians(pdip)
    mid = radians(35.26)
    # print(tdip[0], bdip[0], pdip[0])
    psi = arctan2(sin(tdip), sin(pdip)) - radians(45)
    print(psi)
    # a = cos(35.26) * sin(bdip) * cos(psi)
    a = sin(mid) * cos(bdip) * cos(psi)
    # c = sin(35.26) * sin(bdip)
    b = sin(mid) * sin(bdip) + cos(mid) * sin(bdip) * cos(psi)
    h = cos(bdip) * sin(psi) / b
    v = (cos(mid) * sin(bdip) - a) / b

    return h, v


def radius(theta):
    theta = radians(theta)
    r0 = 6356.752 * 1e3  # WGS84
    r1 = 6378.137 * 1e3
    return sqrt(((r1 ** 2 * cos(theta)) ** 2 + (r0 ** 2 * sin(theta)) ** 2) /
                ((r1 * cos(theta)) ** 2 + (r0 * sin(theta)) ** 2))


def _perturb(mt):
    per = np.zeros_like(mt.mt_err)
    for (i, j), value in np.ndenumerate(mt.mt_err):
        per[i][j] = np.random.normal(scale=value)
    mt_per = MomentTensor(mt.mt + per, mt.exp)

    return mt_per


def simulate_similarity_to_group(mt, mt_group, n=50000):
    Cs = []#np.zeros(n)
    triangle_factor = []#np.zeros(n)
    dips_group = radians(mt_group.axes[:, 1])
    for i in range(n):
        _mt_per = _perturb(mt)
        if _mt_per.fclvd > 0.2:
            continue

        dips = radians(_mt_per.axes[:, 1])

        # triangle_factor[i] = np.sum(np.abs(sin(dips_group) ** 2 - sin(dips) ** 2))
        # Cs[i] = seismic_consistency([_mt_per, mt_group])
        triangle_factor.append(np.sum(np.abs(sin(dips_group) ** 2 - sin(dips) ** 2)))
        Cs.append(seismic_consistency([_mt_per, mt_group]))

    return Cs, triangle_factor


def simulate_uncertainty(mt, n=50000):
    m0_per, f_clvd, axes = [], [], []
    for _ in range(n):
        mt_per = _perturb(mt)
        axes.append(princax(mt_per))
        m0_per.append(mt_per.m0)
        f_clvd.append(mt_per.fclvd)

    return m0_per, f_clvd, np.array(axes)


def simulate_uncertainty_group(tensors, n=50000):
    Cs = []
    for _ in range(n):
        _tensors = []

        for tensor in tensors:
            _tensors.append(_perturb(tensor))

        Cs.append(seismic_consistency(_tensors))

    return Cs


def seismic_consistency(tensors: list):
    # testing_tensors_m0_sum = sum_m0(tensors)
    testing_tensors_m0_sum = len(tensors)  # for any normalised tensor m0 is 1
    testing_tensor_sum_m0 = tensor_sum_normalized(tensors).m0
    Cs = testing_tensor_sum_m0 / testing_tensors_m0_sum

    return Cs


def b_value(tensors, mw_min):
    mws = [tensor.mw for tensor in tensors]
    m = mws[mws >= mw_min]
    return (np.mean(m) - mw_min) * np.log(10)


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


def tensor_sum_normalized(tensors):
    try:
        mt_sum, mt_err_sum = _sum_mt_normalized(tensors)
        exp = log(np.max(mt_sum), 10) // 1
        mt_sum /= 10 ** exp
        mt_err_sum /= 10 ** exp
    except ValueError:
        mt_sum = _sum_mt_normalized(tensors)
        exp = log(np.max(mt_sum), 10) // 1
        mt_sum /= 10 ** exp
        mt_err_sum = None
    # try:
    #     exp = log(np.max(mt_sum), 10) // 1
    # except TypeError:
    #     exp = log(np.max(unumpy.nominal_values(mt_sum)), 10) // 1
    # mt_sum /= 10 ** exp
    # mt_err_sum /= 10 ** exp
    tensor = MomentTensor(mt_sum, exp, mt_err_sum)
    return tensor


def row2mt(data):
    date = datetime.strptime(data[0], '%Y/%m/%d').date()
    pos = (radius(data[1]), data[1], data[2])  # r, lat, lon
    depth = data[3]
    name = data[7]
    exp = data[8]
    mt = np.array([[data[9], data[15], data[17]],
                   [data[15], data[11], data[19]],
                   [data[17], data[19], data[13]]])
    mt_err = np.array([[data[10], data[16], data[18]],
                       [data[16], data[12], data[20]],
                       [data[18], data[20], data[14]]])

    return MomentTensor(mt, exp, mt_err, pos, depth, date, name)


def row2mt_new(data):
    date = data[1]
    pos = (radius(data[2]), data[2], data[3])  # r, lat, lon
    depth = data[4]
    name = data[0]
    exp = data[5]
    mt = np.array(data[6])
    mt_err = np.array(data[7])

    return MomentTensor(mt, exp, mt_err, pos, depth, date, name)


def strain_tensor(tensors, volume, mu=3.3e10):
    return _sum_mt(tensors) / (2 * mu * volume)  # TODO


def plate_vel(tensors, l, w, t, mu=3.3e10):
    return sum_m0(tensors) / (mu * l * w * t)


def _sum_mt(tensors):
    return np.sum([tensor.mt_e for tensor in tensors], axis=0)


def _sum_mt_normalized(tensors):
    # / tensor.m0 to normalise
    mt = np.sum([tensor.mt_e / tensor.m0 for tensor in tensors], axis=0)
    try:
        mt_err = np.sum([tensor.mt_err_e / tensor.m0 for tensor in tensors], axis=0)
        return mt, mt_err
    except AttributeError:
        return mt


def mw2m0(mw):
    return 10 ** (3/2 * (mw + 10.7))


def _m0(mt):
    eigvals, _ = np.linalg.eigh(mt.mt)
    return sqrt(np.sum(eigvals ** 2) / 2) * 10 ** mt.exp


def _mw(mt):
    return 2 / 3 * log(mt.m0, 10) - 10.7


class MomentTensor(object):
    def __init__(self, mt, exp, mt_err=None, pos=None, depth=None, date=None, name=None):
        self.exp = exp
        self.pos = pos
        self.depth = depth
        self.date = date
        self.name = name

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
    def mt_e(self):
        return self.mt * 10 ** self.exp

    @property
    def mt_err_e(self):
        return self.mt_err * 10 ** self.exp

    @property
    def mt6(self):
        return [self.mt[0, 0], self.mt[1, 1], self.mt[2, 2],
                self.mt[0, 1], self.mt[0, 2], self.mt[1, 0]]

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

    @property
    def e_rel(self):
        u = np.linalg.norm(self.mt_err)
        m = np.linalg.norm(self.mt)
        return u / m
