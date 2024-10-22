import numpy as np
'''
Authors: Ted Yeung
Date: Nov 2020
'''

class CavalieriSimpson(object):
    def __init__(self):
        print("CavalieriSimpson")

    @staticmethod
    def integration(y_minus, x_minus, x, x_plus, dt):
        return y_minus + 1/6*(x_minus*dt + 4*x*dt + x_plus*dt)

    @staticmethod
    def di_integrate(data, dt):
        if len(data) > 0:
            cycle_one_forward = data

            vel_right = np.zeros([cycle_one_forward.shape[0], 3])
            # forwards from zero
            for i in range(1, len(cycle_one_forward) - 1):
                for j in range(0, 3):
                    y_minus = float(vel_right[i - 1, j])
                    x_minus = float(cycle_one_forward[i - 1, j])
                    x = float(cycle_one_forward[i, j])
                    x_plus = float(cycle_one_forward[i + 1, j])
                    vel_right[i][j] = CavalieriSimpson.integration(y_minus, x_minus, x, x_plus, dt)
            vel_right[0, :] = np.mean(vel_right[0:2, :], axis=0)
            vel_right[-1, :] = np.mean(vel_right[-2:-1, :], axis=0)

            pos_right = np.zeros([vel_right.shape[0], 3])
            # forwards from zero
            for i in range(1, vel_right.shape[0] - 1):
                for j in range(0, 3):
                    y_minus = float(pos_right[i - 1, j])
                    x_minus = float(vel_right[i - 1, j])
                    x = float(vel_right[i, j])
                    x_plus = float(vel_right[i + 1, j])
                    pos_right[i][j] = CavalieriSimpson.integration(y_minus, x_minus, x, x_plus, dt)
            pos_right[0, :] = np.mean(pos_right[0:2, :], axis=0)
            pos_right[-1, :] = np.mean(pos_right[-2:-1, :], axis=0)
            ret = {
                "vel": vel_right,
                "pos": pos_right
            }
            return ret
        else:
            return None

    @staticmethod
    def ri_integrate(data, dt):
        cycle_one_forward = data
        cycle_one_backward = np.array([cycle_one_forward[i] for i in range(len(cycle_one_forward) - 1, -1, -1)])
        vel_right_bkwd = CavalieriSimpson.di_integrate(cycle_one_backward, dt)
        if vel_right_bkwd is not None:
            vel_right_bkwd["vel"][-1, :] = np.mean(vel_right_bkwd["vel"][-6:-1, :], axis=0)
            vel_right_bkwd["pos"][-1, :] = np.mean(vel_right_bkwd["pos"][-6:-1, :], axis=0)

        return vel_right_bkwd

    @staticmethod
    def dri_weight(bi, di, beta=0.1):
        if bi is not None and di is not None:
            weights = [i / di["pos"].shape[0] for i in range(0, di["pos"].shape[0])]
            ws = WeightingScheme(beta)

            dat_weight = np.zeros(di["pos"].shape)
            for i in range(0, di["pos"].shape[0]-1):
                for j in range(0, 3):
                    dat_weight[i, j] = DRI.filter(bi["pos"][i, j], di["pos"][i, j], ws.w(weights[i]))
            dat_weight[-1, :] = np.mean(dat_weight[-2:-1, :], axis=0)
            return dat_weight
        else:
            return None


class WeightingScheme(object):
    def __init__(self, what_beta=0.1):
        self.beta = what_beta
        self.te = 1  # end time
        self.tb = 0  # begin time

    def set_frame(self, te, tb):
        # need to check if this is needed
        self.te = te
        self.tb = tb

    def s(self, t):
        return np.arctan((1/self.beta)*((2*t-self.te)/2*self.te))

    def w(self, t):
        return (self.s(t)-self.s(self.tb))/(self.s(self.te)-self.s(self.tb))


class DRI(object):
    @staticmethod
    def filter(bi, fi, w):
        return bi*w + fi*(1-w)
