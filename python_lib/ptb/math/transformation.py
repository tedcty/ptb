import json
import os
from typing import Optional, List, Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from util.data import StorageIO, BasicVoxelInfo

'''
Authors: Ted Yeung and Thorben Pauli
Date: Nov 2020
'''


class PCAModel(object):
    def __init__(self):
        self.transformed_data = None
        self.transformation = None

    @staticmethod
    def calculate_eig_and_sort_static(c):
        # Without reduction
        [v, p] = np.linalg.eig(c)
        k = np.sort(v)[::-1]  # descending order
        j = [np.where(v == i)[0][0] for i in k]
        p = p[:, j]
        det_check = np.round(np.linalg.det(p), 10)
        if det_check == -1.0:
            p = PCAModel.handedness_check_static(p)
        return p

    @staticmethod
    def handedness_check_static(p):
        if p.shape[0] == 2 or p.shape[1] == 2:
            return p
        newZ = np.cross(p[:, 0], p[:, 1])
        RH = np.zeros([3, 3])
        RH[:, 0] = p[:, 0]
        RH[:, 1] = p[:, 1]
        RH[:, 2] = newZ
        return RH

    @staticmethod
    def pca_rotation(x):
        nans = np.argwhere(np.isnan(x))
        x0 = x
        if nans.shape[0] != 0:
            x0 = x[:nans[0, 0], :nans[0, 1]]
        n_components = x0.shape[1]
        pcad = PCA(n_components=n_components, svd_solver='full')
        # Transformation with dimension reduction
        zt = pcad.fit_transform(x0)
        mx = pcad.get_covariance()
        mx = PCAModel.calculate_eig_and_sort_static(mx)
        mx = np.transpose(mx)
        p = PCAModel
        p.transformation = mx
        p.transformed_data = zt
        return p


class Quaternion(object):
    def __init__(self, args=None, is_rotation=True, str_precision=3):
        self.str_precision = str_precision
        self.is_null = False
        self.is_rotation = is_rotation
        self.w = 1
        self.i = 0
        self.j = 0
        self.k = 0
        if args is None or len(args) == 0:
            self.w = 1
            self.i = 0
            self.j = 0
            self.k = 0
        elif isinstance(args, list) and len(args) == 4:
            self.w = args[0]
            self.i = args[1]
            self.j = args[2]
            self.k = args[3]
        elif isinstance(args, np.ndarray):
            if args.ndim == 1 and args.shape[0] == 4:
                self.w = args[0]
                self.i = args[1]
                self.j = args[2]
                self.k = args[3]
            elif args.ndim == 2:
                if args.shape[0] == 1 and args.shape[1] == 4:
                    self.w = args[0, 0]
                    self.i = args[0, 1]
                    self.j = args[0, 2]
                    self.k = args[0, 3]
                elif args.shape[0] == 4 and args.shape[1] == 1:
                    self.w = args[0, 0]
                    self.i = args[1, 0]
                    self.j = args[2, 0]
                    self.k = args[3, 0]
                else:
                    print("Warning: input not valid - requires a vector of 4 elements")
                    print("> Default Action: defaulting to identity")
            else:
                print("Warning: input not valid - requires a vector of 4 elements")
                print("> Default Action: defaulting to identity")
        else:
            print("Warning: input not valid - requires a vector of 4 elements")
            print("> Default Action: defaulting to identity")
        self.r = R.from_quat(self.to_array())

    def __repr__(self):
        return "Quaternion"

    def __str__(self):
        ret = str(
            'Quaternion: {:0.' + str(self.str_precision) + 'f} w, {:0.' + str(self.str_precision) + 'f} i, {:0.' + str(
                self.str_precision) + 'f} j, {:0.' + str(self.str_precision) + 'f} k').format(self.w, self.i, self.j,
                                                                                              self.k)
        return ret

    def __sub__(self, other):
        _q = Quaternion()
        _q.w = (self.w - other.w)
        _q.i = (self.i - other.i)
        _q.j = (self.j - other.j)
        _q.k = (self.k - other.k)
        return _q

    def __add__(self, other):
        _q = Quaternion()
        _q.w = (self.w + other.w)
        _q.i = (self.i + other.i)
        _q.j = (self.j + other.j)
        _q.k = (self.k + other.k)
        return _q

    def __mul__(self, _r):
        if self.is_rotation and _r.is_rotation:
            ret: R = self.r * _r.r
            return Quaternion(ret.as_quat())
        else:
            q_ret = self.multiply(_r)
            q_ret.is_rotation = False
            return q_ret

    def __cmp__(self, x):
        w = (self.w - x.w) * (self.w - x.w)
        i = (self.i - x.i) * (self.i - x.i)
        j = (self.j - x.j) * (self.j - x.j)
        k = (self.k - x.k) * (self.k - x.k)

        if (w + i + j + k) < 1.0e-16:
            return True
        else:
            return False

    def multiply(self, r):
        n = Quaternion()
        n.w = r.w * self.w - r.i * self.i - r.j * self.j - r.k * self.k
        n.i = r.w * self.i + r.i * self.w - r.j * self.k + r.k * self.j
        n.j = r.w * self.j + r.i * self.k + r.j * self.w - r.k * self.i
        n.k = r.w * self.k - r.i * self.j + r.j * self.i + r.k * self.w
        return n

    def norm(self):
        a = self.w * self.w
        b = self.i * self.i
        c = self.j * self.j
        d = self.k * self.k
        return np.sqrt(a + b + c + d)

    @staticmethod
    def null():
        _q = Quaternion()
        _q.is_null = True
        _q.w = np.NaN
        _q.i = np.NaN
        _q.j = np.NaN
        _q.k = np.NaN
        return _q

    def to_ypr(self):
        # Method adapted from
        # https:/www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
        sqw = self.w * self.w
        sqx = self.i * self.i
        sqy = self.j * self.j
        sqz = self.k * self.k
        unit = sqx + sqy + sqz + sqw  # if normalised is one, otherwise is correction factor
        test = self.i * self.j + self.k * self.w
        if test > 0.499 * unit:  # singularity at north pole
            heading = 2 * np.arctan2(self.i, self.w)
            attitude = np.pi / 2
            bank = 0
            return [heading, attitude, bank]

        if test < -0.499 * unit:  # singularity at south pole
            heading = -2 * np.arctan2(self.i, self.w)
            attitude = -np.pi / 2
            bank = 0
            return [heading, attitude, bank]

        heading = np.arctan2(2 * self.j * self.w - 2 * self.i * self.k, sqx - sqy - sqz + sqw)
        attitude = np.arcsin(2 * test / unit)
        bank = np.arctan2(2 * self.i * self.w - 2 * self.j * self.k, -sqx + sqy - sqz + sqw)

        return np.array([heading, attitude, bank])

    @staticmethod
    def create_from_ypr(x: list = [0, 0, 0]):
        # Method Adapted from
        # https:/www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
        heading = x[0]
        attitude = x[1]
        bank = x[2]
        c1 = np.cos(heading / 2)
        s1 = np.sin(heading / 2)
        c2 = np.cos(attitude / 2)
        s2 = np.sin(attitude / 2)
        c3 = np.cos(bank / 2)
        s3 = np.sin(bank / 2)
        c1c2 = c1 * c2
        s1s2 = s1 * s2
        w = c1c2 * c3 - s1s2 * s3
        x = c1c2 * s3 + s1s2 * c3
        y = s1 * c2 * c3 + c1 * s2 * s3
        z = c1 * s2 * c3 - s1 * c2 * s3
        return Quaternion([w, x, y, z])

    def conjugate(self):
        n = Quaternion()
        n.w = self.w
        n.i = -self.i
        n.j = -self.j
        n.k = -self.k
        return n

    @staticmethod
    def q_integrate(gy, qp, dt):
        w = Quaternion([0.0, gy[0], gy[1], gy[2]])
        return (qp * w).mult_scale(0.5).mult_scale(dt) + qp

    def inverse(self):
        n = Quaternion()
        n.w = self.w
        n.i = -self.i
        n.j = -self.j
        n.k = -self.k

        w2 = self.w * self.w
        x2 = self.i * self.i
        y2 = self.j * self.j
        z2 = self.k * self.k
        q_sq_sum = w2 + x2 + y2 + z2

        n.w = n.w / q_sq_sum
        n.i = n.i / q_sq_sum
        n.j = n.j / q_sq_sum
        n.k = n.k / q_sq_sum
        return n

    def normalise(self):
        return self / self.norm()

    def unit(self):
        # This is just an alis for normalise
        return self.normalise()

    @staticmethod
    def create_from_helical_angle(x: np.ndarray = np.array([0, 0, 0]), use_scipy=True):
        if use_scipy:
            r = R.from_rotvec(x)
            _q = Quaternion(r.as_quat())
        else:
            n = np.linalg.norm(x)
            nx = np.array(x) / n
            _q = Quaternion()
            _q.w = np.cos(n / 2)
            _q.i = nx[0] * np.sin(n / 2)
            _q.j = nx[1] * np.sin(n / 2)
            _q.k = nx[2] * np.sin(n / 2)
        return _q.normalise()

    def multiply_by_scalar(self, s):
        n = Quaternion()
        n.w = s * self.w
        n.i = s * self.i
        n.j = s * self.j
        n.k = s * self.k
        return n

    def to_array(self):
        return np.array([self.w, self.i, self.j, self.k])

    @staticmethod
    def create_from_euler(seq: str = 'zyx', x: Optional[Union[np.ndarray, List]] = np.array([0, 0, 0]),
                          degrees: bool = False):
        r = R.from_euler(seq, x, degrees=degrees)
        _q = Quaternion(r.as_quat())
        return _q

    def to_euler(self, seq: str = 'zyx', degrees=False):
        r = R.from_quat(self.to_array())
        _m = r.as_euler(seq, degrees=degrees)
        return _m

    @staticmethod
    def create_from_matrix(x=np.eye(3)):
        r = R.from_matrix(x)
        _q = Quaternion(r.as_quat())
        return _q

    def to_mat(self, ):
        r = R.from_quat(self.to_array())
        _m = r.as_matrix()
        return _m

    @staticmethod
    def create_from_rotvec(x=np.array([0, 0, 0])):
        r = R.from_rotvec(x)
        _q = Quaternion(r.as_quat())
        return _q

    def to_rotvec(self):
        r = R.from_quat(self.to_array())
        _m = r.as_rotvec()
        return _m

    def to_axis(self):
        if self.w > 1:
            self.normalise()  # if w>1 acos and sqrt will produce errors, this cant happen if quaternion is normalised

        angle = 2 * np.arccos(self.w)
        # assuming quaternion normalised then w is less than 1, so term always positive.
        s = np.sqrt(1 - self.w * self.w)
        if s < 0.001:  # test to avoid divide by zero, s is always positive due to sqrt
            # if s close to zero then direction of axis not important
            x = self.i  # if it is important that axis is normalised then replace with x=1; y=z=0;
            y = self.j
            z = self.k
        else:
            x = self.i / s  # normalise axis
            y = self.j / s
            z = self.k / s
        return np.array([angle, x, y, z])

    def to_helical(self):
        aq = self.to_axis()
        return np.array([aq[0] * aq[1], aq[0] * aq[2], aq[0] * aq[3]])

    @staticmethod
    def slerp(qa, qb, t):
        # Method Adapted from
        # https:#www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm

        qm = Quaternion()
        # Calculate angle between them
        cosHalfTheta = qa.w * qb.w + qa.i * qb.i + qa.j * qb.j + qa.k * qb.k
        # if qa=qb or qa=-qb then theta = 0 and we can return qa
        if (np.abs(cosHalfTheta) >= 1.000000000000001):
            qm.w = qa.w
            qm.i = qa.i
            qm.j = qa.j
            qm.k = qa.k
            return qm
            # Calculate temporary values.
        halfTheta = np.arccos(cosHalfTheta)
        sinHalfTheta = np.sqrt(1.0 - cosHalfTheta * cosHalfTheta)
        # if theta = 180 degrees then result is not fully defined
        # we could rotate around any axis normal to qa or qb
        if (np.abs(sinHalfTheta) < 0.001):  # # fabs is floating point absolute
            qm.w = (qa.w * 0.5 + qb.w * 0.5)
            qm.i = (qa.i * 0.5 + qb.i * 0.5)
            qm.j = (qa.j * 0.5 + qb.j * 0.5)
            qm.k = (qa.k * 0.5 + qb.k * 0.5)
            return qm

        ratioA = np.sin((1 - t) * halfTheta) / sinHalfTheta
        ratioB = np.sin(t * halfTheta) / sinHalfTheta
        # #calculate Quaternion.
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.i = (qa.i * ratioA + qb.i * ratioB)
        qm.j = (qa.j * ratioA + qb.j * ratioB)
        qm.k = (qa.k * ratioA + qb.k * ratioB)
        return qm


class Trig:
    @staticmethod
    def angle_between_3_points(points=None, p0=None, p1=None, p2=None):
        c = None
        v1 = None
        v2 = None
        if points is not None:
            if isinstance(points, list):
                # assumes list of nd array
                c = points[1]
                v1 = points[0] - c
                v2 = points[2] - c
                pass
            elif isinstance(points, np.ndarray):
                # assumes each column is a marker/point
                c = points[:, 1]
                v1 = points[:, 0] - c
                v2 = points[:, 2] - c
                pass
        elif p0 is not None and p1 is not None and p2 is not None:
            c = p1
            v1 = p0 - c
            v2 = p2 - c
            pass
        if c is not None and v1 is not None and v2 is not None:
            return Trig.angle_between_2_vectors(v1, v2)
        else:
            return None

    @staticmethod
    def angle_between_2_vectors(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class Cloud(object):
    @staticmethod
    def sphere_fit(points):
        p_mean = np.nanmean(points, axis=0)
        n = points.shape[0]
        a = np.eye(3)
        for i in range(0, 3):
            a[i, 0] = np.nansum([(points[x, i] * (points[x, 0] - p_mean[0])) / n for x in range(0, n)])
            a[i, 1] = np.nansum([(points[x, i] * (points[x, 1] - p_mean[1])) / n for x in range(0, n)])
            a[i, 2] = np.nansum([(points[x, i] * (points[x, 2] - p_mean[2])) / n for x in range(0, n)])
        a: np.ndarray = 2 * a
        b: np.ndarray = np.atleast_2d([[0], [0], [0]])
        b[0, 0] = np.nansum(
            [((points[x, 0] ** 2 + points[x, 1] ** 2 + points[x, 2] ** 2) * (points[x, 0] - p_mean[0])) / n for x in
             range(0, n)])
        b[0, 1] = np.nansum(
            [((points[x, 0] ** 2 + points[x, 1] ** 2 + points[x, 2] ** 2) * (points[x, 1] - p_mean[1])) / n for x in
             range(0, n)])
        b[0, 2] = np.nansum(
            [((points[x, 0] ** 2 + points[x, 1] ** 2 + points[x, 2] ** 2) * (points[x, 2] - p_mean[2])) / n for x in
             range(0, n)])
        c = np.matmul(np.linalg.inv(np.matmul(a.transpose(), a)), np.matmul(a.transpose(), b))
        return c

    @staticmethod
    def transform_between_3x3_points_sets(source, target, rowpoints: bool = False, result_as_4x4: bool = True):
        if rowpoints:
            src = source.T
            tar = target.T
        else:
            src = source
            tar = target

        src_m1 = src[:, 0]
        src_m2 = src[:, 1]
        src_m3 = src[:, 2]

        tar_m1 = tar[:, 0]
        tar_m2 = tar[:, 1]
        tar_m3 = tar[:, 2]

        sv1 = src_m2 - src_m1
        sv2 = src_m3 - src_m1
        sv3 = np.cross(sv1, sv2)
        sv2 = np.cross(sv3, sv1)

        nsv1 = np.atleast_2d(sv1 / np.linalg.norm(sv1)).transpose()
        nsv2 = np.atleast_2d(sv2 / np.linalg.norm(sv2)).transpose()
        nsv3 = np.atleast_2d(sv3 / np.linalg.norm(sv3)).transpose()

        tv1 = tar_m2 - tar_m1
        tv2 = tar_m3 - tar_m1
        tv3 = np.cross(tv1, tv2)
        tv2 = np.cross(tv3, tv1)

        ntv1 = np.atleast_2d(tv1 / np.linalg.norm(tv1)).transpose()
        ntv2 = np.atleast_2d(tv2 / np.linalg.norm(tv2)).transpose()
        ntv3 = np.atleast_2d(tv3 / np.linalg.norm(tv3)).transpose()
        # rt = np.hstack((ntv1, ntv2, ntv3))

        t1 = np.matmul(np.array([[1, 0, 0, tar_m1[0]], [0, 1, 0, tar_m1[1]], [0, 0, 1, tar_m1[2]], [0, 0, 0, 1]]),
                       np.vstack((ntv1, [1])))
        t2 = np.matmul(np.array([[1, 0, 0, tar_m1[0]], [0, 1, 0, tar_m1[1]], [0, 0, 1, tar_m1[2]], [0, 0, 0, 1]]),
                       np.vstack((ntv2, [1])))
        t3 = np.matmul(np.array([[1, 0, 0, tar_m1[0]], [0, 1, 0, tar_m1[1]], [0, 0, 1, tar_m1[2]], [0, 0, 0, 1]]),
                       np.vstack((ntv3, [1])))

        s1 = np.matmul(np.array([[1, 0, 0, src_m1[0]], [0, 1, 0, src_m1[1]], [0, 0, 1, src_m1[2]], [0, 0, 0, 1]]),
                       np.vstack((nsv1, [1])))
        s2 = np.matmul(np.array([[1, 0, 0, src_m1[0]], [0, 1, 0, src_m1[1]], [0, 0, 1, src_m1[2]], [0, 0, 0, 1]]),
                       np.vstack((nsv2, [1])))
        s3 = np.matmul(np.array([[1, 0, 0, src_m1[0]], [0, 1, 0, src_m1[1]], [0, 0, 1, src_m1[2]], [0, 0, 0, 1]]),
                       np.vstack((nsv3, [1])))

        a = np.array([[src_m1[0], src_m1[1], src_m1[2], 1],
                      [s1[0, 0], s1[1, 0], s1[2, 0], 1],
                      [s2[0, 0], s2[1, 0], s2[2, 0], 1],
                      [s3[0, 0], s3[1, 0], s3[2, 0], 1]])

        b1 = np.array([[tar_m1[0]], [t1[0, 0]], [t2[0, 0]], [t3[0, 0]]])
        b2 = np.array([[tar_m1[1]], [t1[1, 0]], [t2[1, 0]], [t3[1, 0]]])
        b3 = np.array([[tar_m1[2]], [t1[2, 0]], [t2[2, 0]], [t3[2, 0]]])

        m1 = np.linalg.solve(a, b1)
        m2 = np.linalg.solve(a, b2)
        m3 = np.linalg.solve(a, b3)

        if result_as_4x4:
            m = np.vstack((np.transpose(m1), np.transpose(m2), np.transpose(m3), np.array([0, 0, 0, 1])))
            return m
        else:
            m = np.vstack((np.transpose(m1), np.transpose(m2), np.transpose(m3)))
            return m

    @staticmethod
    def transformation_from_svd(source: np.ndarray,
                                target: np.ndarray,
                                weights: Union[np.ndarray, Iterable] = None,
                                rowpoints: bool = False,
                                result_as_4x4: bool = True):
        """
        Compute the rigid body transformation (rotation and translation) to fit the
        source points to the target points, so that: target = R * source + t
        """
        # Transpose input array if points were given as rows.
        if rowpoints:
            src = source.T
            tar = target.T
        else:
            src = source
            tar = target

        # Check array shape. Given N points, it *must* have the shape 3 x N.
        assert src.shape[0] == 3 and tar.shape[0] == 3
        assert src.shape[1] == tar.shape[1]

        # Check weights.
        n_points = src.shape[1]
        if weights is None:
            w = np.identity(n_points)
        else:
            w = np.diag(weights)
            assert w.shape[0] == w.shape[1] == n_points

        # Compute weighted centroids of both point sets.
        src_mean = (np.matmul(src, w).sum(axis=1) / w.sum()).reshape([3, 1])
        tar_mean = (np.matmul(tar, w).sum(axis=1) / w.sum()).reshape([3, 1])

        # Compute "centralised" points.
        src_c = src - src_mean
        tar_c = tar - tar_mean

        # Compute covariance matrix C = X * W * Y.T
        cov = np.matmul(np.matmul(src_c, w), tar_c.T)

        # Compute singular value decomposition C = U * S * V.T (V.T == vt)
        u, s, vt = np.linalg.svd(cov)

        # Compute rotation matrix R = V * U.T
        r = np.matmul(vt.T, u.T)

        # Compute translation.
        t = tar_mean - np.matmul(r, src_mean)

        if result_as_4x4:
            return np.row_stack([np.column_stack([r, t]), (0, 0, 0, 1)])
        else:
            return r, t

    @staticmethod
    def rigid_body_transform(source, target, rowpoints: bool = False, result_as_4x4: bool = True):
        """
        Compute the rigid body transformation (rotation and translation) to fit the
        source points to the target points, so that: target = R * source + t

        :param source:
        :param target:
        :param rowpoints:
        :param result_as_4x4:
        :return:
        """
        # Check that source and target cloud have the same number of points.
        try:
            assert source.shape == target.shape
        except AssertionError:
            msg = "Input arrays must be the same shape! Got '{0}' and '{1}' instead.".format(source.shape, target.shape)
            raise ValueError(msg)

        if rowpoints:
            src = source.T
            tar = target.T
        else:
            src = source
            tar = target

        # Check array shape. Given N points, it *must* have the shape 3 x N.
        assert src.shape[0] == 3 and tar.shape[0] == 3

        # If three points were given, use hard-coded method.
        if src.shape[1] == 3 and tar.shape[1] == 3:
            return Cloud.transform_between_3x3_points_sets(src, tar, rowpoints=False)

        # If more than three points were given, compute transformation using SVD.
        elif src.shape[1] > 3 and tar.shape[1] > 3:
            return Cloud.transformation_from_svd(src, tar, rowpoints=False, result_as_4x4=True)

        # If less then three points were given, raise error.
        else:
            raise ValueError("Not enough points were given. Requiring at least three points!")

    @staticmethod
    def affine_fit(source, target):
        r = source.shape[1]
        a = np.zeros([4, 4])
        b = np.zeros([4, 3])

        for d in range(0, r):
            b[0, 0] = b[0, 0] + source[0, d] * target[0, d]
            b[1, 0] = b[1, 0] + source[1, d] * target[0, d]
            b[2, 0] = b[2, 0] + source[2, d] * target[0, d]
            b[3, 0] = b[3, 0] + 1 * target[0, d]

            b[0, 1] = b[0, 1] + source[0, d] * target[1, d]
            b[1, 1] = b[1, 1] + source[1, d] * target[1, d]
            b[2, 1] = b[2, 1] + source[2, d] * target[1, d]
            b[3, 1] = b[3, 1] + 1 * target[1, d]

            b[0, 2] = b[0, 2] + source[0, d] * target[2, d]
            b[1, 2] = b[1, 2] + source[1, d] * target[2, d]
            b[2, 2] = b[2, 2] + source[2, d] * target[2, d]
            b[3, 2] = b[3, 2] + 1 * target[2, d]

        for d in range(0, r):
            a[0, 0] = a[0, 0] + source[0, d] * source[0, d]
            a[0, 1] = a[0, 1] + source[0, d] * source[1, d]
            a[0, 2] = a[0, 2] + source[0, d] * source[2, d]
            a[0, 3] = a[0, 3] + source[0, d] * 1

            a[1, 1] = a[1, 1] + source[1, d] * source[1, d]
            a[1, 2] = a[1, 2] + source[1, d] * source[2, d]
            a[1, 3] = a[1, 3] + source[1, d] * 1

            a[2, 2] = a[2, 2] + source[2, d] * source[2, d]
            a[2, 3] = a[2, 3] + source[2, d] * 1

            a[3, 3] = a[3, 3] + 1 * 1

        a[1, 0] = a[0, 1]
        a[2, 0] = a[0, 2]
        a[2, 1] = a[1, 2]
        a[3, 0] = a[0, 3]
        a[3, 1] = a[1, 3]
        a[3, 2] = a[2, 3]

        t = np.transpose(np.matmul(np.linalg.inv(a), b))
        dummy = np.reshape(np.array([0, 0, 0, 1]), [1, 4])
        return np.append(t, dummy, axis=0)

    @staticmethod
    def markerset_to_image_stack(c3d_file: str,
                                 out_dir: str,
                                 filename: str,
                                 para: BasicVoxelInfo = BasicVoxelInfo(),
                                 round_values: int = None,
                                 print_2_screen: bool = False,
                                 print_each_slice: bool = False,
                                 filled_circles: bool = True):
        # Load and read C3D file.
        data = StorageIO.readc3d(c3d_file)
        plt.plot(data['analog'])
        plt.show()
        # Remove time from marker data.
        data["markers"].pop("time")

        # Average marker centres over time and drop marker names.
        markers = [np.nanmean(v, axis=0) for v in data["markers"].values()]

        # Calculate min and max values of marker data, without any additional buffer space, ignoring NaN.
        min_marker = np.nanmin(markers, axis=0)
        max_marker = np.nanmax(markers, axis=0)

        # Get marker radius (mr), and desired bounding box padding (box_pad).
        mr = para.marker_size / 2
        box_pad = para.padding

        # Calculate min and max values for XYZ coordinates, with extra buffer
        # to ensure that all markers are fully enclosed in the space.
        box_mins = min_marker - mr - box_pad
        box_maxs = max_marker + mr + box_pad

        if round_values:
            box_mins = round_down(box_mins, round_values)
            box_maxs = round_up(box_maxs, round_values)

        # Calculate the size ("delta") of the captured space in X, Y, and Z respectively.
        dx = box_maxs[0] - box_mins[0]
        dy = box_maxs[1] - box_mins[1]
        dz = box_maxs[2] - box_mins[2]

        # Calculate required number of slices. This must be an integer value!
        n_slices = int(np.ceil(dz / para.slice_thickness)) + 1
        n_digits = int(np.log10(n_slices)) + 1

        # Since "n_slices" must be an integer, the Z-range must be expanded. Update "box_maxs" accordingly.
        box_maxs[2] = box_mins[2] + (n_slices - 1) * para.slice_thickness

        # Calculate z value for each slice, from z_max to z_min.
        # NOTE: This way, the first page of the tiff file will be to top slice.
        slices_z = np.asarray([box_maxs[2] - i * para.slice_thickness for i in range(n_slices)])

        # Calculate pixel spacing.
        pixel_space = [dx / para.image_size[0], dy / para.image_size[1]]

        # Compose content of info text file, based on calculated values above (json formatted!).
        image_info = {"coord_min": list(box_mins), "coord_max": list(box_maxs), "n_slices": n_slices,
                      "padding": para.padding, "rounded_to": round_values,
                      "img_size": para.image_size, "pixel_spacing": pixel_space,
                      "slice_thickness": para.slice_thickness}

        # Print to screen, if bool was set.
        if print_2_screen:
            print("\nImage Stack Information\n=======================")
            for k in image_info.keys():
                print(k + " = " + str(image_info[k]))
            print()  # For extra emtpy line after the information.

        ################################################################################################################
        # Set parameters for images.
        img_mode = "I;16"
        img_size = para.image_size
        img_colour = 0

        # Create an "image stack" as a list of images.
        img_stack = [Image.new(img_mode, img_size, img_colour) for n in range(n_slices)]

        # Iterate through image stack, and draw each slice individually.
        for i, img in enumerate(img_stack):
            # Get handle to draw on current slice.
            draw = ImageDraw.Draw(img, mode=img_mode)

            # Find all markers that are visible in the current slice.
            visible_markers = [m for m in markers if np.abs(m[2] - slices_z[i]) < mr]

            for m, marker in enumerate(visible_markers):
                # Calculate distance of marker to slice, and radius of intersecting circle.
                dist = np.abs(marker[2] - slices_z[i])
                r = np.sqrt(mr ** 2 - dist ** 2)

                # Calculate the bounding box of the intersecting circle, defined by opposite corner points (in pixel!).
                p0 = [(marker[0] - box_mins[0] - r) / pixel_space[0], (marker[1] - box_mins[1] - r) / pixel_space[1]]
                p1 = [(marker[0] - box_mins[0] + r) / pixel_space[0], (marker[1] - box_mins[1] + r) / pixel_space[1]]

                # Due to differences in coordinate systems between Motion Capture and the Python Image Library (PIL),
                # we need to modify the y-values of the bounding boxes: y_PIL = img_size_in_y - y_MoCap
                p0[1] = img_size[1] - p0[1]
                p1[1] = img_size[1] - p1[1]

                # Since PIL doesn't round floating point pixel values, this needs to be done manually.
                # Additionally, make sure that p0[0] < p1[0] and p0[1] < p1[1].
                cbox = np.asarray([np.round(p0), np.round(p1)])
                cbox = [tuple(x) for x in np.sort(cbox, axis=0)]

                # Draw the intersecting circle on the current slice.
                if filled_circles:
                    draw.ellipse(cbox, fill=255, outline=None)
                else:
                    draw.ellipse(cbox, fill=None, outline=255, width=2)

        # Strip potential file extension of filename.
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            filename = filename.rsplit(".", 1)[0]

        # Make directory, if it doesn't exist.
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # If the flag was set, print each slice individually.
        if print_each_slice:
            for i, img in enumerate(img_stack):
                filepath_img = out_dir + filename + "_{0:0{1}d}.tif".format(i, n_digits)
                img.save(filepath_img, format="TIFF")
            print("Saved {0} individual images to '{1}'.".format(n_slices, out_dir))

        # Save the list of images as an image stack.
        filepath_img = out_dir + filename + ".tif"
        img_stack[0].save(filepath_img, format="TIFF", save_all=True, append_images=img_stack[1:])
        print("Saved image stack to '{}'.".format(filepath_img))

        # Write info text file.
        filepath_txt = out_dir + filename + ".txt"
        with open(filepath_txt, "w") as infofile:
            json.dump(image_info, infofile, indent="\t")
            print("Saved stack infos to '{}'.".format(filepath_txt))


class NpTools:
    @staticmethod
    def repeater(x, n, axis=0):
        if axis == 0:
            ret = np.zeros([x.shape[0] * n, x.shape[1]])
            for i in range(0, n):
                ret[i * x.shape[0]:(i + 1) * x.shape[0], :] = x
            return ret
        elif axis == 1:
            ret = np.zeros([x.shape[0], x.shape[1] * n])
            for i in range(0, n):
                ret[:, i * x.shape[1]:(i + 1) * x.shape[1]] = x
            return ret
        return None


def round_down(x, d):
    return x - np.mod(x, d)


def round_up(x, d):
    return x + np.mod(-np.asarray(x), d)


if __name__ == '__main__':
    # Quick Tests
    test_markerset_to_image_stack = False
    test_quaternion = False
    # data = StorageIO.readc3d_general("../../resources/c3d_test/Latency_0_Correction_01.c3d")
    data = StorageIO.readc3d_general("../../resources/markerset_to_image_stack_test/input/XSHD00001_crossCal.c3d")
    plt.figure()
    plt.plot(data['analog_data']['EMGs.1'])
    plt.show()
    pass
    if test_markerset_to_image_stack:
        Cloud.markerset_to_image_stack(
            c3d_file="../../resources/markerset_to_image_stack_test/input/XSHD00001_crossCal.c3d",
            out_dir="../../resources/markerset_to_image_stack_test/output/",
            filename="XSHD00001_crossCal_v01")

    if test_quaternion:
        q = Quaternion(np.array([[1], [0.5], [0.2], [0]]))
        print(q)
        q = Quaternion(np.array([[1, 0.5, 0.2, 0]]))
        print(q)
        q = Quaternion(np.array([1, 0.5, 0.2, 0]))
        print(q)
        q = Quaternion([1, 0.5, 0.2, 0])
        print(q)
        q = Quaternion()
        print(q)
        q = Quaternion([])
        print(q)
        q = Quaternion(np.array([]))
        print(q)
        q = Quaternion(np.array([0, 1, 2, 3, 4, 5]))
        print(q)

        q0 = Quaternion.create_from_euler('zyx', [90, 90, 45], True)
        print(q)
        q1 = Quaternion.null()
        print(q)
