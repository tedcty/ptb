import io
import numpy as np
import os
import json
import struct
import time
import zipfile
import gzip
import pickle

from xml.dom import minidom
from copy import deepcopy
from enum import Enum

import pandas as pd
from scipy.stats import mode
from tqdm import tqdm

from ptb.core import Yatsdo
from ptb.util.io.mocap.file_formats import TRC
from ptb.util.io.mocap.low_lvl import c3d as c3d

from subprocess import Popen, PIPE
from os import path


class StorageType(Enum):
    unknown = [-1, 'unknown']
    nexus = [0, 'csv']
    mot = [1, 'mot']
    C3D = [3, 'c3d']
    csv = [4, 'csv']
    captureU = [5, 'csv']
    trc = [6, 'trc']
    sto = [7, 'sto']

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

    def write_mot(self, filename):
        lines = []
        for h in self.info['header']:
            lines.append(h.strip()+"\n")
        cols = ""
        for c in self.data.columns:
            cols += c
            cols += "\t"
        lines.append(cols.strip()+"\n")
        self.data.to_csv(filename+".csv", sep="\t", index=False)
        f = open(filename+".csv", "r")
        count = 0
        stream = "Hello world"
        while len(stream) > 0:
            stream = f.readline()
            if count == 0:
                count += 1
                continue
            else:
                count += 1
                lines.append(stream)
        f.close()

        with open(filename, 'w') as writer:
            writer.writelines(lines)
        pass

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
    def scan_nexus_for_device_data(filename):
        """
        This method currently only tested on outputs from VICON Nexus 2.12
        :param filename: CSV file output from VICON Nexus 2.12
        :return:
        """
        f = open(filename, "r")
        buffer = []
        start_id = -100
        end_id = 0
        ids = 0
        stream = "hoi, allo, moshi moshi?"
        boo = True
        boo1 = True
        meta_ = {'Device Sampling Rate': 0, 'Devices Columns': [], 'Device Unit': {}}
        device_boo = False
        unit_boo = True
        data_buffer = []
        data_frame = 0
        device_end = False
        other_data = []
        while len(stream) > 0:
            stream = f.readline()
            if unit_boo:
                buffer.append(stream)
            if device_boo:
                meta_['Device Sampling Rate'] = float(stream.strip())
                device_boo = False

            if stream.find("Device") > -1 and not device_boo:
                device_boo = True

            if stream.find("Frame") > -1 and boo:
                device_labels = buffer[len(buffer)-2].split(',')
                data_labels = buffer[len(buffer)-1].split(',')
                current_device = ""
                for d in range(0, len(device_labels)):
                    if len(device_labels[d]) > 0:
                        current_device = device_labels[d] + " - "
                        if '\n' in current_device:
                            break
                    meta_['Devices Columns'].append(current_device+data_labels[d])

                start_id = ids + 1
                boo = False
            if ids == start_id:
                dunits = buffer[-1].split(',')
                meta_['Device Unit'] = {meta_['Devices Columns'][dc]: dunits[dc] for dc in range(0, len(meta_['Devices Columns']))}
                unit_boo = False
                pass
            cleaned = stream.strip()
            if ids >= start_id + 1 and not unit_boo:
                elements = cleaned.split(',')
                try:
                    if int(elements[0]) >= data_frame:
                        data_frame = int(elements[0])
                        data_buffer.append(np.asfarray(cleaned.split(',')))
                except ValueError:
                    device_end = True
            if device_end:
                other_data.append(cleaned)
            if len(cleaned) == 0 and boo1 and not boo:
                boo1 = False
                end_id = ids - 1
            ids += 1
        f.close()

        if ids > 0:
            temp = [n for n in data_buffer if n.shape[0] == len(meta_['Devices Columns'])]
            data_np = np.zeros([len(temp), len(meta_['Devices Columns'])])
            for n in range(0, len(temp)):
                data_np[n, :] = temp[n]
            data_df = pd.DataFrame(data=data_np, columns=meta_['Devices Columns'])
            return [i for i in range(0, start_id)], [start_id, end_id], meta_, data_df
        else:
            return [], [], meta_, None

    @staticmethod
    def trc_reader(filename, delimiter="\t", headers=True, fill_data=False):
        """
        This method reads in a trc file.
        :param filename: string path to file
        :param delimiter: how to parse the components, default is tab spaces
        :param headers: Save Header in Object, default is True
        :param fill_data: fill missing frame data, if you have gaps in your trc file used this otherwise trc file will not load
        :return: TRC (Yatdos) object containing the information in the TRC file
        """
        return TRC.read(filename, delimiter, headers, fill_data)

    @staticmethod
    def simple_read(filename, delimit=False, delimiter=","):
        stream = "hoi"
        buffer = []
        try:
            f = open(filename, "r")
            while len(stream) > 0:
                stream = f.readline()
                if delimit:
                    elements = stream.strip().split(delimiter)
                    buffer.append(elements)
                else:
                    buffer.append(stream)
        except OSError:
            print('cannot open', filename)
        finally:
            f.close()
        return buffer

    @staticmethod
    def simple_write(data, filename, delimit=False, delimiter=","):
        try:
            lines = data
            if delimit:
                lines = []
                for b in data:
                    line = ""
                    for i in range(0, len(b)):
                        line += b[i]
                        if i < len(b):
                            line += delimiter
                    line += "\n"
                    lines.append(line)
            with open(filename, 'w') as writer:
                writer.writelines(lines)

        except OSError:
            print('cannot open', filename)

    @staticmethod
    def failsafe_omega(filename, delimiter=",", headers=True):
        """
        If this doesn't work then the file is not a text file
        :param filename:
        :param delimiter:
        :param headers:
        :return:
        """
        ret = []
        lengths = []
        stream = "hoi"
        read_successful = False
        try:
            f = open(filename, "r")
            while len(stream) > 0:
                stream = f.readline()
                elements = stream.strip().split(delimiter)
                ret.append(elements)
                lengths.append(len(elements))
            read_successful = True
        except OSError:
            print('cannot open', filename)
        finally:
            f.close()
        if read_successful:
            col = mode(lengths)[0][0]
            bounty = [b for b in range(0, len(lengths)) if lengths[b] != col]
            ret2 = [ret[i] for i in range(0, len(ret)) if i not in bounty]
            if headers:
                header = ret2.pop(0)
            try:
                data = [[float(i[j]) if j > 0 else int(i[j]) for j in range(0, col)] for i in ret2]
                return pd.DataFrame(data=data, columns=header)
            except ValueError:
                print("Why")
        return None

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
    def mot_header_getter(filename):
        f = open(filename, "r")
        stream = "hoi"
        index = 0
        ret = []
        while len(stream) > 0:
            stream = f.readline()
            ret.append(stream)
            if stream.find("endheader") > -1:
                break
            index += 1
        f.close()
        return ret

    @staticmethod
    def file_extension_check(path):
        filename, file_extension = os.path.splitext(path)
        return StorageType.check_for_known(file_extension)

    @staticmethod
    def readc3d_general(filename):
        with open(filename, 'rb') as c3dfile:
            reader = c3d.Reader(c3dfile)
            first_frame = reader.first_frame()
            units = reader.groups.get('POINT').get('UNITS').bytes.decode("utf-8")
            subject_ = reader.groups.get('SUBJECTS')
            if subject_ is None:
                subject = {'name': "",
                           'marker_sets': "",
                           'is_static': ""}
            else:
                subject = {'name': subject_.params['NAMES'].bytes.decode("utf-8").strip(),
                           'marker_sets': subject_.params['MARKER_SETS'].bytes.decode("utf-8").strip(),
                           'is_static': int.from_bytes(subject_.params['IS_STATIC'].bytes, 'little')}
            trial_ = reader.groups.get('TRIAL')
            t0 = int.from_bytes(trial_.params['ACTUAL_START_FIELD'].bytes, 'little')
            t1 = int.from_bytes(trial_.params['ACTUAL_END_FIELD'].bytes, 'little')
            ccr = None
            try:
                cam_rate = trial_.params['CAMERA_RATE'].bytes

                ccr = struct.unpack('f', cam_rate)[0]
            except KeyError:
                pass
            trial = {'camera_rate': ccr,
                     'start_frame': t0,
                     'end_frame': t1}
            num_frames = reader.last_frame() - first_frame
            frames = []
            fp_ = reader.groups.get('FORCE_PLATFORM')
            used_byte = fp_.params['USED'].bytes
            used = (np.frombuffer(used_byte, dtype=np.int16))[0]
            channel = fp_.params['CHANNEL'].int16_array
            origin = fp_.params['ORIGIN'].float_array
            corners = fp_.params['CORNERS'].float_array
            corners_dimension = fp_.params['CORNERS'].dimensions
            corners2 = fp_.params['CORNERS'].bytes
            corners2_bytes_per_float = fp_.params['CORNERS'].bytes_per_element
            if corners2_bytes_per_float == 4:
                dtyp = np.float32
            else:
                dtyp = np.float64
            corners2_float_array = np.frombuffer(corners2, dtype=dtyp)
            corners2_float_array = np.squeeze(corners2_float_array)

            c_fp = np.reshape(corners2_float_array, corners_dimension, order='F')
            force_plate_corners = {'force_plate_{0}'.format(i + 1): c_fp[:, :, i] for i in range(0, used)}
            origins = []
            for i in force_plate_corners:
                origins.append(np.mean(force_plate_corners[i], axis=1))
            try:
                analog_labels = [a.strip() for a in reader.analog_labels]
            except AttributeError:
                analog_labels = []
            point_labels = [a.strip() for a in reader.point_labels]
            rate_diff = int(reader.analog_rate / reader.point_rate)
            analog_group = reader.groups.get('ANALOG')
            try:
                analog_unit = analog_group.params['UNITS'].string_array
            except KeyError:
                analog_unit = []
            except UnicodeDecodeError:
                bs = []
                for a in analog_group.params['UNITS'].bytes_array:
                    b = a.decode('latin-1').strip()
                    bs.append(b)
                analog_unit = bs
            if len (analog_labels) == len(analog_unit):
                analog_units = {analog_labels[a]:analog_unit[a] for a in range(0, len(analog_unit))}
            ret = {'analog_channels_label': analog_labels,
                   'number_of_force_plates': used,
                   'force_plate_corners': force_plate_corners,
                   'force_plate_corners_np': corners,
                   'force_plate_origins_from_corner': origins,
                   'force_plate_origins': origin,
                   'force_plate_channels': channel,
                   'num_analog_channels': len(analog_labels),
                   'num_frames': num_frames,
                   'first_frame': first_frame,
                   'num_analog_frames': int((num_frames + 1) * rate_diff),
                   'analog_unit': analog_units,
                   'rate-diff(A2M)': rate_diff,
                   'point_label': point_labels,
                   'point_label_expanded': 0,
                   'num_points': len(reader.point_labels),
                   'point_rate': reader.point_rate,
                   'analog_rate': reader.analog_rate,
                   'point_unit': units,
                   'meta_data': {
                       'subject': subject,
                       'trial': trial},
                   'mocap': {"rates": reader.point_rate, "units": units, "header": reader.header}
                   }

            flatten = lambda t: [item for sublist in t for item in sublist]
            expanded_labels = []
            if len(ret['point_label']) > 0:
                expanded_labels = [[label + '_X', label + '_Y', label + '_Z'] for label in point_labels]
                expanded_labels = flatten(expanded_labels)

            if ret['num_analog_channels'] > 0:
                analog_data = np.zeros([(first_frame + num_frames) * int(rate_diff), ret['num_analog_channels'] + 2])
            else:
                analog_data = None
            if ret['num_points'] > 0:
                point_data = np.zeros([num_frames + first_frame, ret['num_points'] * 3])
            else:
                point_data = None

            for i, points, analog in reader.read_frames():
                if analog_data is not None:
                    try:
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 2:] = analog.transpose()
                    except ValueError:
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 2:] = analog
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 1] = [j for j in
                                                                                                  range(0, rate_diff)]
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 0] = i
                if point_data is not None:
                    xyz = points[:, :3]
                    point_data[(i - 1), :] = xyz.reshape([1, ret['num_points'] * 3])
                frames.append({"id": i, "points": points, "analog": analog})

            times = [d*(1/ret['point_rate']) for d in range(0, point_data.shape[0])]
            pt_data = np.zeros([point_data.shape[0], point_data.shape[1]+1])
            pt_data[:, 0] = times
            pt_data[:, 1:] = point_data
            expanded_labels.insert(0, 'time')
            df = pd.DataFrame(data=pt_data, columns=expanded_labels)
            cols_analog = ["mocap-frame", "sub-frame"]
            for c in ret['analog_channels_label']:
                cols_analog.append(c)
            df_analog = pd.DataFrame(data=analog_data, columns=cols_analog)
            ret['point_data'] = df
            ret['analog_data'] = df_analog
            ret['data'] = frames
        return ret

    @staticmethod
    def simple_readc3d(filename, exportas_text=False):
        with open(filename, 'rb') as c3dfile:
            reader = c3d.Reader(c3dfile)
            units = reader.groups.get('POINT').get('UNITS').bytes.decode("utf-8")
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
                if isinstance(analog, np.ndarray):
                    if len(analog.shape) > 1:
                        analog_temp = pd.DataFrame(data=analog, columns=reader.analog_labels)
                        # analogs.append(analog[:, 0])
                        # This contains more than 1 frame as there is usually 5 frames per 1 frame of point data if
                        # data OMC is recorded at 200 Hz and analog data recorded at 1000 Hz
                        analogs.append(analog_temp)
                markers[marker_labels[0]].append((i - 1) * (1 / reader.point_rate))
                for j in range(1, len(marker_labels)):
                    errors = points[j - 1, 3:]
                    p = points[j - 1, 0:3]
                    for e in errors:
                        if e == -1:
                            version = np.__version__.split('.')
                            if len(version) == 3:
                                if int(version[0].strip()) < 2:
                                    p = np.asarray([np.NaN, np.NaN, np.NaN])
                                else:
                                    p = np.asarray([np.nan, np.nan, np.nan])
                            break
                    markers[marker_labels[j]].append(p)
                if exportas_text:
                    ret += 'frame {}: point {}, analog {}'.format(
                        i, points, analog)
            box = {'markers': markers, 'mocap': {"rates": reader.point_rate, "units": units, "header": reader.header},
                   'text': ret, "analog": analogs}
        return box

    @staticmethod
    def load(filename, sto_type: StorageType = None, fill_data=False):
        if sto_type is None:
            sto_type = StorageType.check_for_known(filename)
        sto = StorageIO()
        skip_rows = []
        delimiter = ' '
        data = None
        is_trc = False
        if sto_type == StorageType.nexus:
            skip_rows, st_en, meta_data, data = StorageIO.scan_nexus_for_device_data(filename)
            delimiter = ','
            sto.info["devices"] = meta_data
        elif sto_type == StorageType.mot or sto_type == StorageType.sto:
            skip_rows = StorageIO.mot_header(filename)
            sto.info["header"] = StorageIO.mot_header_getter(filename)
            delimiter = '\t'
        elif sto_type == StorageType.csv or sto_type == StorageType.captureU:
            delimiter = ','
        elif sto_type == StorageType.trc:
            delimiter = '\t'
            is_trc = True
        trc = None
        try:
            if sto_type == StorageType.nexus:
                p = data
                time_list = [i*1/meta_data['Device Sampling Rate'] for i in range(0, p.shape[0])]
                p.insert(2, 'time', time_list)
            if sto_type == StorageType.trc:
                trc = StorageIO.trc_reader(filename, delimiter=delimiter, fill_data=fill_data)
                p = trc.to_panda()
            if (sto_type == StorageType.mot or sto_type == StorageType.sto or
                    sto_type == StorageType.csv or sto_type == StorageType.captureU):
                p = pd.read_csv(filename, delimiter=delimiter, skiprows=skip_rows, index_col=False, dtype=np.float64)
                pass

        except pd.errors.ParserError:
            p = StorageIO.failsafe_omega(filename, delimiter=delimiter)
            if p is None:
                return None
        except IOError:
            p = StorageIO.failsafe_omega(filename, delimiter=delimiter)
            if p is None:
                return None
        except IndexError:
            p = StorageIO.failsafe_omega(filename, delimiter=delimiter)
            if p is None:
                return None
        col = p.columns
        drops = [c for c in col if 'Unnamed' in c]
        q = p.drop(columns=drops)
        sto.buffer.append(q)
        drops = [c for c in col if 'unix_timestamp_microsec' in c]
        if len(drops) > 0:
            r = p.drop(columns=drops)
            sto.buffer.append(r)
        try:
            sto.find_dt(sto_type)
        except TypeError:
            sto.info['dt'] = 0.01
        if not is_trc:
            return sto
        else:
            return trc

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

    def find_dt(self, st):
        # a helper function for a special case where get_first column is time
        t = 0
        if st == StorageType.trc:
            t = 1
        df = self.data.to_numpy()
        df0 = df[:-1, t]
        df1 = df[1:, t]
        dt = -1
        if df.shape[0] > 1:
            dt = np.round(np.nanmean(df1 - df0), 5)
            self.info['dt'] = dt
        return dt

    @staticmethod
    def load_json(file):
        return JSONSUtl.load_json(file)

    @staticmethod
    def write_json(pathto, dictm):
        return JSONSUtl.write_json(pathto, dictm)

    @property
    def dt(self):
        try:
            return self.info['dt']
        except KeyError:
            self.find_dt(StorageType.mot)

class Kirby:
    def __init__(self, data):
        """
        This helper object compresses/ decompress other objects and export and load it as a gzip file
        """
        self.data = data

    @staticmethod
    def load(file_pathx):
        with gzip.open(file_pathx, "rb") as fx:
            loaded_data = pickle.load(fx)
        return loaded_data

    def export(self, file_pathx):
        with gzip.open(file_pathx, 'wb', compresslevel=9) as fx:
            pickle.dump(self, fx)
        pass




class Bapple:
    def __init__(self, x: Yatsdo = None, y: Yatsdo = None):
        """
        Version 2 of the Bapple (simplified):
        This Bapple just holds the data, export and import using npz instead of csv
        Hopefully this is faster
        :param x: Measurement/ Input data (i.e. IMU) (Type: pandas Dataframe)
        :param y: Measurement/ Input data (i.e. Target) (Type: pandas Dataframe)
        """
        self.__apples__ = x
        self.__bananas__ = y
        self.__filename__ = ""
        self.__filepath__ = ""

    @property
    def apples(self):
        """
        This creates a property of the class call apples, which allow you to access the apples (x)
        :return: apples
        """
        return self.__apples__

    @apples.setter
    def apples(self, a, para=None):
        if isinstance(a, np.ndarray) and para is not None:
            self.__apples__ = Yatsdo(a, para[MinionKey.columns])
        elif isinstance(a, pd.DataFrame) or isinstance(a, np.ndarray):
            self.__apples__ = Yatsdo(a)

    @property
    def bananas(self):
        """
        This creates a property of the class call bananas, which allow you to access the bananas (y)
        :return: banana
        """
        return self.__bananas__

    @bananas.setter
    def bananas(self, b, para=None):
        if isinstance(b, np.ndarray) and para is not None:
            self.__bananas__ = Yatsdo(b, para[MinionKey.columns])
        elif isinstance(b, pd.DataFrame) or isinstance(b, np.ndarray):
            self.__bananas__ = Yatsdo(b)

    @staticmethod
    def load(filepath: str):
        """
        This method direct loads bapple into memory
        :param filepath:
        :return:
        """
        start = time.time()
        block = np.load(filepath)
        meta_data = json.load(io.BytesIO(block['meta_data.json']))
        print("Loading apples and bananas")
        data_block = None
        a_block = None
        b_block = None
        try:
            data_block = np.load(io.BytesIO(block['data.npz']))
        except KeyError:
            a_block = np.frombuffer(block['a.npz'])
            a_block = a_block.reshape(meta_data['apple']['shape'])
            b_block = np.frombuffer(block['b.npz'])
            b_block = b_block.reshape(meta_data['banana']['shape'])

        print("Preparing the Bapple")
        a_bapple = None
        b_bapple = None

        if data_block is not None:
            a_bapple = pd.DataFrame(data=np.asarray(data_block['a']), columns=meta_data['apple'])
            b_bapple = pd.DataFrame(data=np.asarray(data_block['b']), columns=meta_data['banana'])
        if a_block is not None:
            a_bapple = pd.DataFrame(data=a_block, columns=meta_data['apple']['cols'])
        if b_block is not None:
            b_bapple = pd.DataFrame(data=b_block, columns=meta_data['banana']['cols'])

        bar = tqdm(total=3, desc="Cooking", ascii=False,
                   ncols=120,
                   colour="#6e5b5b")
        bar.update(2)
        bapple = Bapple(x=Yatsdo(a_bapple), y=Yatsdo(b_bapple))
        bar.update(1)
        bar.close()
        end = time.time()
        print("Order up: One Bapple > Load completed in {0:0.2f}s".format(end - start))
        return bapple

    def export(self, filename: str, pathto: str, compresslevel=1):
        """
        Export bapple to a zip file
        Note: May fail if there is not enough space for the temp folder
        :param filename:
        :param pathto:
        :return:
        """
        start = time.time()
        if not pathto.endswith("/") or not pathto.endswith("\\"):
            pathto += "/"

        meta_data = {}

        if not os.path.exists(pathto+"temp/"):
            os.makedirs(pathto+"temp/")

        num_files = 0
        if self.apples is not None:
            num_files += 1
        if self.bananas is not None:
            num_files += 1
        num_files += 1

        with zipfile.ZipFile(pathto + filename, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as archive:
            bar = tqdm(total=num_files, desc="Saving to {0} >".format(filename), ascii=False,
                       ncols=120,
                       colour="#6e5b5b")
            if self.apples is not None and self.bananas is not None:
                bar.desc = "Saving to {0} > Exporting Apples and Bananas".format(filename)
                bar.refresh()
                meta_data['apple'] = {'cols': [c for c in self.apples.column_labels], 'shape': self.apples.data.shape}
                meta_data['banana'] = {'cols': [c for c in self.bananas.column_labels], 'shape': self.bananas.data.shape}
                anp = self.apples.data
                bnp = self.bananas.data

                archive.writestr("a.npz", anp.tobytes())
                bar.update(1)
                archive.writestr("b.npz", bnp.tobytes())
                bar.update(1)

            bar.desc = "Saving to {0} > Exporting meta_data.json".format(filename)
            bar.refresh()
            json_string = json.dumps(meta_data)
            meta_data_bytes = json_string.encode('utf-8')
            archive.writestr("meta_data.json", meta_data_bytes)
            bar.update(1)
            bar.close()
        end = time.time()
        print("Order up: One Bapple > Saved completed in {0:0.2f}s".format(end - start))
        pass


class MinionKey(Enum):
    butterworth = 0
    cutoff = 1
    sampling_freq = 2
    order = 3
    apple = 4
    banana = 5
    bapple = 6
    by_time = 7
    by_index = 8
    chop_type = 8.1
    all = 10
    columns = 11
    size = 12
    data = 13
    start = 14
    increment = 15
    dt = 16
    combined = 17
    id = 18

    @staticmethod
    def get_by_name(name):
        for v in MinionKey:
            if v.name == name:
                return v
        return None


class BasicVoxelInfo:
    # This is data holder
    def __init__(self, slice_thickness=0.625, image_size=[512, 512]):
        self.slice_thickness = slice_thickness
        self.image_size = image_size
        self.padding = 10
        self.marker_size = 4

class MYXML:
    # This is a simple xml reader/ writer
    def __init__(self, filename):
        self.tree = minidom.parse(filename)

    def set_value(self, tag, value):
        a = self.tree.getElementsByTagName(tag)
        print("Before: {0}".format(a[0].childNodes[0].data))
        a[0].childNodes[0].data = value
        print("After: {0}".format(a[0].childNodes[0].data))
        pass

    def add_node(self, parent, new_node, value):
        x = self.tree.getElementsByTagName(parent)
        a = self.tree.createElement(new_node)
        b = self.tree.createTextNode(value)
        a.appendChild(b)
        x[0].appendChild(a)

    def add_value(self, node, value):
        x = self.tree.getElementsByTagName(node)
        b = self.tree.createTextNode(value)
        x[0].appendChild(b)

    def write(self, filename, pretty=False):
        myfile = open(filename, "w")
        if pretty:
            dom_str = self.tree.toprettyxml()
            dom_str_element = [f+"\n" for f in dom_str.split("\n") if len(f.strip()) > 0]
            dom_str_new = ""
            dom_str_new = dom_str_new.join(dom_str_element)
            myfile.write(dom_str_new)
        else:
            myfile.write(self.tree.toxml())
        myfile.close()


class JSONSUtl:
    def __init__(self):
        print("Json Helper")
    @staticmethod
    def write_json(pathto, dictm, indent=4, sort_keys=True):
        with open(pathto, 'w') as outfile:
            json.dump(dictm, outfile, sort_keys=sort_keys, indent=indent)

    @staticmethod
    def load_json(file):
        with open(file, 'r') as infile:
            data = json.load(infile)
        return data

class BasicIO:
    @staticmethod
    def read_txt(filename):
        with open(filename) as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def read_as_block(filename):
        ret = ""
        for s in BasicIO.read_txt(filename):
            ret += s + "\n"
        return ret

'''
This module contains methods dealing with os operations
i.e.
Getting a list of hard drive names etc
'''


def list_harddrives():
    """
    This function returns a table of hard drives info:
    'DeviceID', 'VolumeName', 'Description'

    Note: currently this is hardcoded
    :return: p, Panda Dataframe of hard drives in/connected to the PC
    """
    # Assumes fix column size
    wmic_path = "C:/Windows/System32/WMIC.exe"
    if not path.exists(wmic_path):
        wmic_path = "C:/Windows/System32/wbem/WMIC.exe"

    if path.exists(wmic_path):
        drives = os.popen(wmic_path+" logicaldisk get deviceid, volumename, description").readlines()
        lines = []
        boo = True
        di = 0
        vn = 0
        for d in drives:
            if len(d) > 1:
                lc = d[:-1].strip()
                if boo:
                    di = lc.index("DeviceID")
                    vn = lc.index("VolumeName")
                    boo = False

                la = [lc[:di].strip(), lc[di:vn].strip(), lc[vn:].strip()]
                lines.append(la)
        p = pd.DataFrame(data=lines[1:], columns=lines[0])
    elif path.exists('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'):
        print("Trying alternative powershell method ...")
        pipe = Popen(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', 'Get-Volume'], stdin=PIPE, stdout=PIPE,
                  stderr=PIPE)
        output, err = pipe.communicate()
        drives = output.decode("utf-8")
        drs = drives.strip().splitlines()
        columns_index = drs[1].split(' ')
        columns_size = [len(columns_index[i]) for i in range(len(columns_index))]
        columns_id = [0]
        current = 0
        for i in range(len(columns_size)):
            columns_id.append(columns_size[i] + current)
            current = current + columns_size[i] + 1
        col_names = [(drs[0][columns_id[i]:columns_id[i+1]]).strip() for i in range(len(columns_id)-1)]
        lines = []
        for j in range(2, len(drs)):
            lines.append([(drs[j][columns_id[i]:columns_id[i+1]]).strip() for i in range(len(columns_id)-1)])
        p = pd.DataFrame(data=lines, columns=col_names)
        p.rename(columns={'DriveLetter': 'DeviceID', 'FriendlyName': 'VolumeName'}, inplace=True)
    else:
        p = None
    if p is not None:
        col_new_order = ['DeviceID', 'VolumeName', 'Description']
        p = p.reindex(columns=col_new_order)
        print()
    return p

def drive(volume_name):
    d = list_harddrives()
    ids = d[d['VolumeName'] == volume_name].index.values.astype(int)[0]
    d_letter = d["DeviceID"].iloc[ids]
    if ":" not in d_letter:
        d_letter = d_letter+":"
    return d_letter