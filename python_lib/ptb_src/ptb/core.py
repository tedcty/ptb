import io
import json
import math
import os
import time
import zipfile
from enum import Enum

import importlib.util

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

from ptb.util.math.filters import Butterworth


class info(Enum):
    name = "PTB"
    version = "0.3.13"
    dscr = "\nAuckland Bioengineering Institute's Musculoskeletal Modelling Group\nPresents . . .\nA Python Toolbox\n"
    all = ""

    def __str__(self):
        ret = "{0}".format(self.value)
        if self == info.all:
            ret = "{0}\nVersion: {1}\n{2}".format(info.name, info.version, info.dscr)
        return ret

    @staticmethod
    def is_version(version: str):
        if str(info.version) == version:
            return True
        return False

class AdditionalPackages:
    @staticmethod
    def install_gias3():
        pkg_list = {
            "pydicom:": 'python -m pip install pydicom==2.4.4',
            "gias3":'python -m pip install gias3',
            "gias3.musculoskeletal": 'python -m pip install gias3.musculoskeletal',
            "gias3.io": 'python -m pip install gias3.io',
            "gias3.mapclientpluginutilities": 'python -m pip install gias3.mapclientpluginutilities',
            "gias3.visualisation": 'python -m pip install gias3.visualisation',
            "gias3.applications": 'python -m pip install gias3.applications',
            "gias3.examples": 'python -m pip install gias3.examples'
        }
        for g in pkg_list:
            if g == 'opensim':
                if importlib.util.find_spec("opensim") is not None:
                    continue
            os.system(pkg_list[g])


class Yatsdo(object):
    @staticmethod
    def create_from_storage_io(s, para=None):
        cols = [c for c in s.data.columns]
        ret = np.zeros(s.data.shape)
        t = np.array(s.data['time'].to_list())
        ret[:, 0] = t
        cut = 5
        fs = 1 / s.dt
        order = 4
        if para is not None:
            try:
                b = para['butter']
                cut = b['cut']
                fs = b['fs']
                order = b['order']
            except KeyError:
                pass
        for c in range(1, len(cols)):
            k = np.array(s.data[cols[c]].to_list())
            k_ret = Butterworth.butter_low_filter(k, cut, fs, order)
            ret[:, c] = k_ret
        ret_pd = pd.DataFrame(data=ret, columns=cols)
        return Yatsdo(ret_pd)

    def to_panda_data_frame(self):
        ret = pd.DataFrame(data=self.data, columns=self.column_labels)
        return ret
    # Yet another time series data object
    #
    # This object take a 2D np array and creates
    # functions for simple data manipulation
    # i.e. resampling data
    # Assume get_first column is time
    def __init__(self, data, col_names=[], fill_data=False, time_col=0):
        self.col_labels = col_names
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data
        elif isinstance(data, pd.DataFrame):
            self.data: np.ndarray = data.to_numpy()
            self.col_labels = [c for c in data.columns]
        else:
            self.data: np.ndarray = np.array(data)
        self.curve = {}
        self.dev_curve = {}
        self.x = self.data[:, time_col]
        if self.data.shape[0] > 3:
            for i in range(1, self.data.shape[1]):
                p = interpolate.InterpolatedUnivariateSpline(self.x, self.data[:, i])
                a = np.isnan(self.data[:, i])
                boo = np.sum(a) > 0 # is there nan
                b = [b for b in range(0, a.shape[0]) if not a[b]]
                if len(b) < a.shape[0] and fill_data or boo:
                    try:
                        p = interpolate.InterpolatedUnivariateSpline(self.x[b], self.data[b, i])
                    except Exception:
                        pass
                self.curve[i] = p
                if p is not None:
                    self.dev_curve[i] = p.derivative()

        self.size = self.data.shape
        pass

    @staticmethod
    def load(filepath: str):
        """
        This method direct loads bapple into memory
        :param filepath:
        :param sep: path sep
        :param unzip: set to unzip or not default False
        :param del_temp:
        :return:
        """
        start = time.time()
        bar = tqdm(total=5, desc="Cooking", ascii=False,
                   ncols=120,
                   colour="#6e5b5b")
        block = np.load(filepath)
        bar.update(1)
        meta_data = json.load(io.BytesIO(block['meta_data.json']))
        bar.update(1)
        print("Loading apples and bananas")
        data_block = None
        a_block = None
        try:
            data_block = np.load(io.BytesIO(block['data.npz']))
        except KeyError:
            a_block = np.frombuffer(block['a.npz'])
            a_block = a_block.reshape(meta_data['pineapple']['shape'])
        bar.update(1)
        print("Preparing the pineapple")
        a_bapple = None
        if data_block is not None:
            a_bapple = pd.DataFrame(data=np.asarray(data_block['a']), columns=meta_data['pineapple'])

        if a_block is not None:
            a_bapple = pd.DataFrame(data=a_block, columns=meta_data['pineapple']['cols'])
        bar.update(1)
        bapple = Yatsdo(a_bapple)
        bar.update(1)
        bar.close()
        end = time.time()
        print("Order up: One Yatdos > Load completed in {0:0.2f}s".format(end - start))
        return bapple

    def export(self, filename: str, pathto: str, compresslevel=1):
        """
        Export bapple to a zip file
        Note: May fail if there is not enough space for the temp folder
        :param filename:
        :param pathto:
        :return:
        """

        if not pathto.endswith("/") or not pathto.endswith("\\"):
            pathto += "/"

        meta_data = {}

        if not os.path.exists(pathto+"temp/"):
            os.makedirs(pathto+"temp/")

        num_files = 0
        if self.data is not None:
            num_files += 1
        num_files += 1

        with zipfile.ZipFile(pathto + filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as archive:
            bar = tqdm(total=num_files, desc="Saving to {0} >".format(filename), ascii=False,
                       ncols=120,
                       colour="#6e5b5b")
            if self.data is not None:
                bar.desc = "Saving to {0} > Exporting Pineapple".format(filename)
                bar.refresh()
                meta_data['pineapple'] = {'cols': [c for c in self.column_labels], 'shape': self.data.shape}
                anp = self.data

                archive.writestr("a.npz", anp.tobytes())
                bar.update(1)

            bar.desc = "Saving to {0} > Exporting meta_data.json".format(filename)
            bar.refresh()
            json_string = json.dumps(meta_data)
            meta_data_bytes = json_string.encode('utf-8')
            archive.writestr("meta_data.json", meta_data_bytes)
            bar.update(1)
            bar.close()

    @property
    def shape(self):
        return self.data.shape

    @property
    def dt(self):
        """
        Get the mean sampling rate of the data
        :return: sampling rate (float)
        """
        a = 0.01    # default
        if self.x.shape[0] > 2:
            a = np.nanmean(self.x[1:] - self.x[:-1])
        return a

    # Given time points it will get the accompanying data
    def get_samples(self, time_points, assume_time_first_col=True, as_pandas=False):
        a = [time_points]
        # additional_col = 0
        # if not assume_time_first_col:
        #     additional_col = 1
        # cols = self.data.shape[1] + additional_col
        cols = self.data.shape[1]
        rows = len(time_points)
        ret = np.zeros([rows, cols])
        if not assume_time_first_col:
            ret[:, 0] = [i for i in range(1, rows+1)]
            ret[:, 1] = time_points
            for i in range(2, self.data.shape[1]):
                a.append(self.curve[i](time_points))
                ret[:, i] = self.curve[i](time_points)
        else:
            ret[:, 0] = time_points
            for i in range(1, self.data.shape[1]):
                a.append(self.curve[i](time_points))
                ret[:, i] = self.curve[i](time_points)

        # return np.squeeze(np.array(a)).transpose()

        if as_pandas:
            if len(self.column_labels) > 0:
                col = self.column_labels
                return pd.DataFrame(data=ret, columns=col)
            else:
                return pd.DataFrame(data=ret)

        return ret

    def get_sample(self, time_point, assume_time_first_col=True):
        additional_col = 0
        if not assume_time_first_col:
            additional_col = 1
        cols = self.data.shape[1] + additional_col
        ret = np.zeros([1, cols])
        ret[:, 0] = time_point
        for i in range(1, self.data.shape[1]):
            ret[:, i] = self.curve[i](time_point)
        return ret[0]

    # Given time points it will get the accompanying data
    def get_samples_dev(self, time_points):
        a = [time_points]
        for i in range(1, self.data.shape[1]):
            a.append(self.dev_curve[i](time_points))
        return np.squeeze(np.array(a)).transpose()

    def update(self):
        self.x = self.data[:, 0]
        for i in range(1, self.data.shape[1]):
            p = interpolate.InterpolatedUnivariateSpline(self.x, self.data[:, i])
            self.curve[i] = p

    def update_Spline(self):
        for i in range(1, self.data.shape[1]):
            p = interpolate.InterpolatedUnivariateSpline(self.x, self.data[:, i])
            self.curve[i] = p

    def filter(self, func, para):
        to_filter = self.data
        for i in range(1, to_filter.shape[1]):
            d = to_filter[:, i]
            to_filter[:, i] = func(np.squeeze(d), para)
        return Yatsdo(pd.DataFrame(data=to_filter, columns=self.column_labels))

    def remove_jitter(self):
        rate = 1 / np.nanmean(self.x[1:] - self.x[0:-1])
        period = self.x[-1] - self.x[0]
        frames = period / (1 / rate)
        new_frame = self.get_samples([self.x[0] + (1 / rate) * i for i in range(0, int(math.ceil(frames)))])
        return Yatsdo(pd.DataFrame(data=new_frame, columns=self.column_labels))

    @staticmethod
    def combine(a, b, rate=None):
        if rate is None:
            rate_a = 1 / np.nanmean(a.x[1:] - a.x[0:-1])
            rate_b = 1 / np.nanmean(b.x[1:] - b.x[0:-1])
            rate = rate_a
            if rate < rate_b:
                rate = rate_b
        period = a.x[-1] - a.x[0]
        frames = period / (1 / rate)
        dfR = a.get_samples([a.x[0] + (1 / rate) * i for i in range(0, int(math.ceil(frames)))])
        period = b.x[-1] - b.x[0]
        frames = period / (1 / rate)
        dfR1 = b.get_samples([b.x[0] + (1 / rate) * i for i in range(0, int(math.ceil(frames)))])
        lab = [c for c in a.column_labels]
        for c in range(1, len(b.column_labels)):
            lab.append(b.column_labels[c])
        return Yatsdo.merge(dfR, dfR1, lab)

    @staticmethod
    def merge(a, b, cols_ab, ignore_time=True):
        """
        This method merges two nd_arrays together and returns a Yatsdo
        :param a: nd_array
        :param b: nd_array
        :param cols_ab: list
        :param ignore_time: boolean
        :return: Yatsdo
        """
        rows = a.shape[0]
        if b.shape[0] < rows:
            rows = b.shape[0]
        cols = a.shape[1]+b.shape[1]-1
        nova = np.zeros([rows, cols])
        if ignore_time:
            nova[:rows, 0:a.shape[1]] = a[:rows, :]
            nova[:rows, a.shape[1]:cols] = b[:rows, 1:]
        y = Yatsdo(nova)
        y.column_labels = cols_ab
        return y

    @property
    def column_labels(self):
        return self.col_labels

    @column_labels.setter
    def column_labels(self, x):
        self.col_labels = x

    def to_csv(self, file_path):
        d = pd.DataFrame(data=self.data, columns=self.column_labels)
        d.to_csv(file_path, index=False)
