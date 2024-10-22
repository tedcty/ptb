from scipy import interpolate
from yatpkg.math.filters import Butterworth
import numpy as np
import pandas as pd
from enum import Enum
import os
import yatpkg.util.c3d as c3d
import vtk
import math
from copy import deepcopy

'''
Authors: Ted Yeung
Date: Nov 2020
'''


class Yatsdo:
    # Yet another time series data object
    #
    # This object take a 2D np array and creates
    # functions for simple data manipulation
    # i.e. resampling data
    # Assume get_first column is time
    def __init__(self, data, col_names=[]):
        self.col_labels = col_names
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data
        elif isinstance(data, pd.DataFrame):
            self.data: np.ndarray = data.to_numpy()
            self.col_labels = [c for c in data.columns]
        else:
            self.data: np.ndarray = np.array(data)
        self.curve = {}
        self.x = self.data[:, 0]
        for i in range(1, self.data.shape[1]):
            p = interpolate.InterpolatedUnivariateSpline(self.x, self.data[:, i])
            self.curve[i] = p
        self.size = self.data.shape
        pass

    # Given time points it will get the accompanying data
    def get_samples(self, time_points):
        a = [time_points]
        for i in range(1, self.data.shape[1]):
            a.append(self.curve[i](time_points))
        return np.squeeze(np.array(a)).transpose()

    def update(self):
        self.x = self.data[:, 0]
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


class Yadict(object):
    # Yet another dictionary
    # provides core set and get dict
    # and get keys in a list
    def __init__(self, d: dict):
        self.d = d

    def __getitem__(self, item):
        return self.d[item]

    def __setitem__(self, item, value):
        self.d[item] = value

    def get_keys(self):
        return [k for k in self.d]


class IMU(object):
    '''
    Will need to restructure
    '''
    def __init__(self, acc: Yatsdo, gyro: Yatsdo, mag=None, ori=None):
        self.acc = acc
        self.gyr = gyro
        self.mag = mag
        self.ori = ori
        self.ori_filtered = True

    def butterworth_filter(self, cut_off, sampling_rate):
        for d in range(1, self.acc.size[1]):
            data = self.acc.data[:, d]
            filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
            self.acc.data[:, d] = filtered
        self.acc.update()

        for d in range(1, self.gyro.size[1]):
            data = self.gyro.data[:, d]
            filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
            self.gyro.data[:, d] = filtered
        self.gyro.update()

        if self.mag is not None:
            for d in range(1, self.mag.size[1]):
                data = self.mag.data[:, d]
                filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
                self.mag.data[:, d] = filtered
            self.mag.update()

        if self.ori is not None and not self.ori_filtered:
            for d in range(1, self.ori.size[1]):
                data = self.mag.ori[:, d]
                filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
                self.ori.data[:, d] = filtered
            self.ori.update()

    def export(self, filename, boo=[True, True, False, True]):
        imu = [self.acc, self.gyr, self.mag, self.ori]
        ret = np.zeros([self.size[0], 1+(np.sum(boo)*3)])
        columns = ['time']
        boos = ['acc', 'gyr', 'mag', 'ori']
        ret[:, 0] = self.acc.x
        col = 1
        for i in range(0, len(boos)):
            if boo[i]:
                columns.append(boos[i]+'_x')
                columns.append(boos[i]+'_y')
                columns.append(boos[i]+'_z')
                ret[:, col: col+3] = imu[i].data[:, 1:]
                col = col + 3

        p = pd.DataFrame(data=ret, columns=columns)
        p.to_csv(filename, index=False)


class StorageType(Enum):
    unknown = [-1, 'unknown']
    nexus = [0, 'csv']
    mot = [1, 'mot']
    C3D = [3, 'c3d']
    csv = [4, 'csv']

    @staticmethod
    def check_for_known(ext):
        for st in StorageType:
            s = st.value
            if s[1] in ext.lower():
                return st
        return StorageType.unknown


class StorageIO(object):
    def __init__(self, store: pd.DataFrame = None, buffer_size: int = 10):
        # A storage object that helps read (mot or csv with known formats) and write csv
        self.buffer = []
        self.buffer_size = buffer_size
        if store is not None:
            self.buffer.append(store)
        self.info = {}

    @property
    def data(self):
        # returns last loaded
        return deepcopy(self.buffer[-1])

    @data.setter
    def data(self, x):
        if len(self.buffer) >= 10:
            self.buffer.pop(0)
        self.buffer.append(x)

    @staticmethod
    def find_nexus_header(filename):
        f = open(filename, "r")
        start_id = 0
        end_id = 0
        ids = 0
        stream = "hoi"
        boo = True
        boo1 = True
        while len(stream) > 0:
            stream = f.readline()
            if stream.find("Frame") > -1 and boo:
                start_id = ids + 2
                boo = False
            cleaned = stream.strip()
            if len(cleaned) == 0 and boo1 and not boo:
                boo1 = False
                end_id = ids + 2
            ids += 1
        f.close()

        if ids > 0:
            return [i for i in range(0, start_id)], [start_id, end_id]
        else:
            return []

    @staticmethod
    def mot_header(filename):
        f = open(filename, "r")
        stream = "hoi"
        index = 0
        while len(stream) > 0:
            stream = f.readline()
            if stream.find("endheader") > -1:
                break
            index += 1
        f.close()
        if index > 0:
            return [i for i in range(0, index + 1)]
        else:
            return []

    @staticmethod
    def file_extension_check(path):
        filename, file_extension = os.path.splitext(path)
        return StorageType.check_for_known(file_extension)

    @staticmethod
    def readc3d_general(filename):
        with open(filename, 'rb') as c3dfile:
            reader = c3d.Reader(c3dfile)
            first_frame = reader.first_frame()
            num_frames = reader.last_frame()-first_frame
            frames = []
            analog_labels = [a.strip() for a in reader.analog_labels]

            ret = {'analog_channels_label': analog_labels,
                   'num_analog_channels': len(analog_labels),
                   'num_frames': num_frames,
                   'num_analog_frames': int((num_frames + 1)*(reader.analog_rate/reader.point_rate)),
                   'point_label': reader.point_labels,
                   'point_label_expanded': 0,
                   'num_points': len(reader.point_labels),
                   'point_rate': reader.point_rate}

            flatten = lambda t: [item for sublist in t for item in sublist]
            expanded_labels = []
            if len(ret['point_label']) > 0:
                expanded_labels = [[label+'_X', label+'_Y', label+'_Z'] for label in reader.point_labels]
                expanded_labels = flatten(expanded_labels)
                print()

            if ret['num_analog_channels'] > 0:
                analog_data = np.zeros([ret['num_analog_frames']+(first_frame+num_frames), ret['num_analog_channels']])
            else:
                analog_data = None
            if ret['num_points'] > 0:
                point_data = np.zeros([num_frames+first_frame, ret['num_points']*3])
            else:
                point_data = None

            for i, points, analog in reader.read_frames():
                if analog_data is not None:
                    analog_data[(i-1)*10: ((i-1)*10)+10, :] = analog.transpose()
                if point_data is not None:
                    xyz = points[:, :3]
                    point_data[(i-1), :] = xyz.reshape([1, ret['num_points']*3])
                frames.append({"id": i, "points": points, "analog": analog})
            df = pd.DataFrame(data=point_data, columns=expanded_labels)
            df_analog = pd.DataFrame(data=analog_data, columns=ret['analog_channels_label'])
            ret['point_data'] = df
            ret['analog_data'] = df_analog
            ret['data'] = frames
        return ret

    @staticmethod
    def readc3d(filename, exportas_text=False):
        with open(filename, 'rb') as c3dfile:
            reader = c3d.Reader(c3dfile)
            ret = ""
            markers = {'time': []}
            count = 0
            marker_labels = ['time']
            for pl in reader.point_labels:
                lb = pl.strip()
                if len(lb) == 0:
                    lb = 'M{:03d}'.format(count)
                    count += 1
                marker_labels.append(lb)
                markers[lb] = []
            analogs = []
            for i, points, analog in reader.read_frames():
                analogs.append(analog[:, 0])
                markers[marker_labels[0]].append((i-1)*(1/reader.point_rate))
                for j in range(1, len(marker_labels)):
                    errors = points[j-1, 3:]
                    p = points[j-1, 0:3]
                    for e in errors:
                        if e == -1:
                            p = np.asarray([np.NaN, np.NaN, np.NaN])
                            break
                    markers[marker_labels[j]].append(p)
                if exportas_text:
                    ret += 'frame {}: point {}, analog {}'.format(
                        i, points, analog)
            box = {'markers': markers, 'mocap': {"rates": reader.point_rate}, 'text': ret, "analog": analogs}
        return box

    @staticmethod
    def load(filename, sto_type: StorageType = None):
        if sto_type is None:
            sto_type = StorageType.check_for_known(filename)
        sto = StorageIO()
        skip_rows = []
        delimiter = ' '
        if sto_type == StorageType.nexus:
            skip_rows = StorageIO.find_nexus_header(filename)
            delimiter = ','
        elif sto_type == StorageType.mot:
            skip_rows = StorageIO.mot_header(filename)
            delimiter = '\t'
        elif sto_type == StorageType.csv:
            delimiter = ','
        p = pd.read_csv(filename, delimiter=delimiter, skiprows=skip_rows)
        col = p.columns
        drops = [c for c in col if 'Unnamed' in c]
        q = p.drop(columns=drops)
        sto.buffer.append(q)
        drops = [c for c in col if 'unix_timestamp_microsec' in c]
        if len(drops) > 0:
            r = p.drop(columns=drops)
            sto.buffer.append(r)

        sto.find_dt()
        return sto

    def to_csv(self, path):
        self.data.to_csv(path, index=False)

    def to_yatsdo(self, remove_jitter=True):
        y = Yatsdo(self.data)
        if remove_jitter:
            return y.remove_jitter()
        return y

    def separate(self, parts, first_col=None):
        if len(parts) == 0:
            return {"imu": self.data}
        if len(parts) == 1:
            return {parts[0]: self.data}
        data_headings = self.data.columns
        separated_part = {}
        for b in parts:
            part = []
            if first_col is not None:
                part = [first_col]
            for c in data_headings:
                if b in c.lower():
                    part.append(c)
            separated_part[b] = self.data[part]
        return separated_part

    def find_dt(self):
        # a helper function for a special case where get_first column is time
        df = self.data.to_numpy()
        df0 = df[:-1, 0]
        df1 = df[1:, 0]
        dt = np.round(np.nanmean(df1 - df0), 5)
        self.info['dt'] = dt
        return dt


class BasicVoxelInfo:
    def __init__(self, slice_thickness=0.625, image_size=[512, 512]):
        self.slice_thickness = slice_thickness
        self.image_size = image_size
        self.padding = 10
        self.marker_size = 4


class Mesh:
    @staticmethod
    def convert_vtp_2_stl(vtp_file, stl_out):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_file)
        reader.Update()

        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetFileName(stl_out)
        stl_writer.SetInputConnection(reader.GetOutputPort())
        stl_writer.Write()


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


if __name__ == '__main__':
    pass

