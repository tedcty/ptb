import numpy
import numpy as np
import pandas

from util.data import Yatsdo


class Stat(object):
    @staticmethod
    def center_data(x: np.ndarray):
        # calculate mean and center data to mean
        u = np.atleast_2d(np.sum(x, axis=0) / x.shape[0])
        u = u.transpose()
        um = np.repeat(u.transpose(), x.shape[0], axis=0)
        return x - um, np.squeeze(u)

    @staticmethod
    def covariance_matrix(x: np.ndarray):
        return (1 / x.shape[0]) * np.dot((np.transpose(x)), x)

    @staticmethod
    def eig_rh(x):
        v, p = np.linalg.eig(x)
        k = np.sort(v)[::-1]  # descending order
        j = [np.where(v == i)[0][0] for i in k]
        p0 = p[:, j]
        det_ = np.round(np.linalg.det(p0), 10)
        if det_ == -1.0:
            new_z = np.cross(p0[:, 0], p0[:, 1])
            rh = np.zeros([3, 3])
            rh[:, 0] = p0[:, 0]
            rh[:, 1] = p0[:, 1]
            rh[:, 2] = new_z
            p0 = rh
        return k, p0


def resample(data, target_freq):
    yoda = data
    # type check
    if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
        yoda = Yatsdo(data)
    elif not isinstance(data, Yatsdo):
        return None
    start = yoda.x[0]
    end = yoda.x[-1]
    cs = np.round((end - start) / (1.0 / target_freq), 5) + 1
    tp = [np.round(start + c * (1 / target_freq), 5) for c in range(0, int(cs))]
    d = yoda.get_samples(tp)
    if isinstance(data, pd.DataFrame):
        d = pd.DataFrame(data=d, columns=data.columns)
    elif isinstance(data, Yatsdo):
        return yoda
    return d
