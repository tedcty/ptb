import io
import json
import os
import struct
import time
import zipfile
from copy import deepcopy
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm

from util.data import Yatsdo
from util.io.mocap.file_formats import TRC
from util.io.mocap.low_lvl import c3d as c3d


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
            subject = {'name': subject_.params['NAMES'].bytes.decode("utf-8").strip(),
                       'marker_sets': subject_.params['MARKER_SETS'].bytes.decode("utf-8").strip(),
                       'is_static': int.from_bytes(subject_.params['IS_STATIC'].bytes, 'little')}
            trial_ = reader.groups.get('TRIAL')
            t0 = int.from_bytes(trial_.params['ACTUAL_START_FIELD'].bytes, 'little')
            t1 = int.from_bytes(trial_.params['ACTUAL_END_FIELD'].bytes, 'little')
            trial = {'camera_rate': struct.unpack('f', trial_.params['CAMERA_RATE'].bytes)[0],
                     'start_frame': t0,
                     'end_frame': t1}
            num_frames = reader.last_frame()-first_frame
            frames = []
            analog_labels = [a.strip() for a in reader.analog_labels]
            point_labels = [a.strip() for a in reader.point_labels]
            rate_diff = int(reader.analog_rate/reader.point_rate)
            ret = {'analog_channels_label': analog_labels,
                   'num_analog_channels': len(analog_labels),
                   'num_frames': num_frames,
                   'first_frame': first_frame,
                   'num_analog_frames': int((num_frames + 1)*rate_diff),
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
                expanded_labels = [[label+'_X', label+'_Y', label+'_Z'] for label in point_labels]
                expanded_labels = flatten(expanded_labels)

            if ret['num_analog_channels'] > 0:
                analog_data = np.zeros([(first_frame+num_frames)*int(rate_diff), ret['num_analog_channels'] + 2])
            else:
                analog_data = None
            if ret['num_points'] > 0:
                point_data = np.zeros([num_frames+first_frame, ret['num_points']*3])
            else:
                point_data = None

            for i, points, analog in reader.read_frames():
                if analog_data is not None:
                    try:
                        analog_data[(i-1)*rate_diff: ((i-1)*rate_diff)+rate_diff, 2:] = analog.transpose()
                    except ValueError:
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 2:] = analog
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 1] = [j for j in range(0, rate_diff)]
                        analog_data[(i - 1) * rate_diff: ((i - 1) * rate_diff) + rate_diff, 0] = i
                if point_data is not None:
                    xyz = points[:, :3]
                    point_data[(i-1), :] = xyz.reshape([1, ret['num_points']*3])
                frames.append({"id": i, "points": points, "analog": analog})
            df = pd.DataFrame(data=point_data, columns=expanded_labels)
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
            box = {'markers': markers, 'mocap': {"rates": reader.point_rate, "units": units, "header": reader.header}, 'text': ret, "analog": analogs}
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
            if sto_type == StorageType.mot or sto_type == StorageType.csv or sto_type == StorageType.captureU:
                p = pd.read_csv(filename, delimiter=delimiter, skiprows=skip_rows)

        except pd.errors.ParserError:
            p = StorageIO.failsafe_omega(filename, delimiter=delimiter)
            if p is None:
                return None
        except IOError:
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
