import numpy as np

'''
Authors: Ted Yeung
Date: Nov 2020
'''

class DataError:
    @ staticmethod
    def rms(x: np.ndarray, y: np.ndarray):
        y = np.atleast_2d(y)
        diff = (x - y)
        ds_mean = np.nanmean(diff ** 2)
        rms = np.sqrt(ds_mean)
        return rms

    @staticmethod
    def mae(x: np.ndarray, y: np.ndarray):
        y = np.atleast_2d(y.to_numpy())
        diff = np.abs(x - y)
        ds_mean = np.nanmean(diff)
        return ds_mean
