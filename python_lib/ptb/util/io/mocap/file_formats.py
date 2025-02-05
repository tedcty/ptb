import copy
import os
from copy import deepcopy
from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from util.data import Yatsdo
from util.io.helper import StorageIO, StorageType


class TRC(Yatsdo):

    @staticmethod
    def read(filename, delimiter="\t", headers=True, fill_data=False):
        """
        This method reads in a trc file.
        Note - This should be moved to the TRC object
        :param filename:
        :param delimiter:
        :param headers:
        :param fill_data:
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
        meta_info = {ret[1][i]: ret[2][i] for i in range(0, len(ret[1]))}
        headings = [c for c in ret[3] if len(c) > 1]
        d = {}
        i = 0
        markers_col = [r for r in ret[4] if len(r) > 1]
        for c in headings[2:]:
            d[c] = {"col": markers_col[i:i + 3]}
            i += 3
        data_headings = ["Frame#", "Time"]
        for dh in d:
            dhl = [dh + "_" + h for h in d[dh]["col"]]
            for j in dhl:
                data_headings.append(j)
        st = 5
        start_data = ret[st]
        while len(start_data) < 2:
            st += 1
            start_data = ret[st]
        ed = st
        for k in range(st, len(ret)):
            if len(ret[k]) < 2:
                break
            ed += 1
        data = []
        cols = len(data_headings)

        for n in ret[st:ed]:
            dd = []
            for m in n:
                try:
                    dd.append(float(m))
                except ValueError:
                    dd.append(np.NaN)
            while len(dd) < cols:
                dd.append(np.NaN)
            data.append(dd)
        rows = len(data)
        np_data = np.zeros([rows, cols])
        for i in range(0, np_data.shape[0]):
            dx = data[i]
            if len(dx) == cols:
                np_data[i, :] = dx
        trc = TRC(np_data, data_headings, filename, fill_data=fill_data)
        trc.headers = meta_info
        trc.raw = ret
        trc.marker_names = headings[2:]
        trc.st = st
        k = 0
        try:
            for m in trc.marker_names:
                trc.marker_set[m] = pd.DataFrame(trc.data[:, 2 + k:5 + k], columns=d[m]['col'])
                k += 3
            if fill_data:
                trc.fill_gaps()
        except AssertionError:
            print("error creating {0} marker".format(m))
            cols = d[m]['col']
            error_data = trc.data[:, 2 + k:5 + k]
            pass
        # trc.write("G:/Shared drives/Rocco Hip Replacement Study/01/Pre-op/Sit00a.trc")
        return trc

    def get_samples_as_trc(self, time_points):
        subset = self.get_samples(time_points, assume_time_first_col=False)
        trc = copy.deepcopy(self)
        trc.data = subset
        trc.x = subset[:, 1]
        trc.update()
        return trc

    def get_data_between_timepoints(self, starttime, endtime, freq):
        period = endtime - starttime
        dt = 1.0/(1.0*freq)
        num_frames = period / dt
        timepoints = [t * dt + starttime for t in range(0, int(num_frames))]
        subset = self.get_samples_as_trc(timepoints)
        return subset

    def __init__(self, data, col_names=[], filename="", fill_data=False):
        super().__init__(data, col_names, fill_data, time_col=1)
        self.marker_names = []
        self.marker_set = {}
        self.raw = None
        self.st = 0
        self.first_line = "PathFileType\t3\t(X/Y/Z)\t{0}".format(filename)
        self.headers_labels = [MocapFlags.DataRate,
                               MocapFlags.CameraRate,
                               MocapFlags.NumFrames,
                               MocapFlags.NumMarkers,
                               MocapFlags.Units,
                               MocapFlags.OrigDataRate,
                               MocapFlags.OrigDataStartFrame,
                               MocapFlags.OrigNumFrames]
        self.headers = {h: -1.0 for h in self.headers_labels}
        if data is not None:
            self.x = self.data[:, 1]

    def to_panda(self):
        return pd.DataFrame(data=self.data, columns=self.column_labels)

    @property
    def dt(self):
        """
        Get the mean sampling rate of the data
        :return: sampling rate (float)
        """
        a = 0.01  # default
        if self.x.shape[0] > 2:
            a = np.nanmean(self.x[1:] - self.x[:-1])
        return a

    def fill_gaps(self):
        n = self.get_samples(self.x)
        self.data[:, 2:] = n[:, 2:]
        self.update()
        pass


    @staticmethod
    def create_from_c3d_dict(c3d_dic, filename, fill_data=False):
        header = c3d_dic["mocap"]["header"]
        trc_header = {k.value: -1 for k in MocapFlags.defaults_to_list()}
        trc_header[MocapFlags.DataRate.value] = header.frame_rate
        trc_header[MocapFlags.CameraRate.value] = header.frame_rate
        trc_header[MocapFlags.NumFrames.value] = header.last_frame
        trc_header[MocapFlags.Units.value] = MocapFlags.unit(c3d_dic['mocap']['units']).value

        trc_header[MocapFlags.OrigDataStartFrame.value] = header.first_frame
        trc_header[MocapFlags.OrigDataRate.value] = header.frame_rate
        trc_header[MocapFlags.OrigNumFrames.value] = header.last_frame
        markers = c3d_dic["markers"]
        markers_np = {n: np.array(markers[n]) for n in markers}
        marker_labels = [m for m in markers_np]
        trc_header[MocapFlags.NumMarkers.value] = len(markers) - 1
        frames_block = np.zeros([header.last_frame, (3 * len(marker_labels)) + 1])
        col_names = ['Frame#', 'time']
        inx = 1
        frames_block[:, 0] = [int(n + 1) for n in range(0, header.last_frame)]
        times = markers_np[marker_labels[0]]
        indx = 0
        if np.nanmin(markers_np[marker_labels[0]]) < 0:
            for t in range(0, times.shape[0]):
                indx = t
                if times[t] >= 0.0:
                    break
            times = times[indx:]
        frames_block[:, 1] = times
        markers_set = {}
        for m in range(1, len(marker_labels)):
            lb = marker_labels[m]
            m_np = markers_np[lb]
            s = (m * 3 + 1) - 2
            e = (m * 3 + 4) - 2
            frames_block[:, s: e] = m_np[indx:, :]
            col_names.append(marker_labels[m] + '_X{0}'.format(inx))
            col_names.append(marker_labels[m] + '_Y{0}'.format(inx))
            col_names.append(marker_labels[m] + '_Z{0}'.format(inx))
            markers_set[marker_labels[m]] = pd.DataFrame(data=m_np, columns=['X{0}'.format(inx), 'Y{0}'.format(inx),
                                                                             'Z{0}'.format(inx)])
            inx += 1

        trc = TRC(frames_block, col_names=col_names, filename=filename[:filename.rindex(".")] + ".trc", fill_data=fill_data)
        trc.headers = trc_header
        trc.marker_set = markers_set
        trc.marker_names = marker_labels[1:]
        return trc

    @staticmethod
    def create_from_c3d(data, fill_data=False):
        c3d_data = StorageIO.simple_readc3d(data)
        return TRC.create_from_c3d_dict(c3d_data, data)
        # header = c3d_data["mocap"]["header"]
        # trc_header = {k.value: -1 for k in MocapFlags.defaults_to_list()}
        # trc_header[MocapFlags.DataRate.value] = header.frame_rate
        # trc_header[MocapFlags.CameraRate.value] = header.frame_rate
        # trc_header[MocapFlags.NumFrames.value] = header.last_frame
        # trc_header[MocapFlags.Units.value] = MocapFlags.unit(c3d_data['mocap']['units']).value
        #
        # trc_header[MocapFlags.OrigDataStartFrame.value] = header.first_frame
        # trc_header[MocapFlags.OrigDataRate.value] = header.frame_rate
        # trc_header[MocapFlags.OrigNumFrames.value] = header.last_frame
        # markers = c3d_data["markers"]
        # markers_np = {n: np.array(markers[n]) for n in markers}
        # marker_labels = [m for m in markers_np]
        # trc_header[MocapFlags.NumMarkers.value] = len(markers)-1
        # frames_block = np.zeros([header.last_frame, (3 * len(marker_labels))+1])
        # col_names = ['Frame#', 'time']
        # inx = 1
        # frames_block[:, 0] = [int(n+1) for n in range(0, header.last_frame)]
        # frames_block[:, 1] = markers_np[marker_labels[0]]
        # markers_set = {}
        # for m in range(1, len(marker_labels)):
        #     lb = marker_labels[m]
        #     m_np = markers_np[lb]
        #     s = (m * 3 + 1) - 2
        #     e = (m * 3 + 4) - 2
        #     frames_block[:, s: e] = m_np
        #     col_names.append(marker_labels[m] + '_X{0}'.format(inx))
        #     col_names.append(marker_labels[m] + '_Y{0}'.format(inx))
        #     col_names.append(marker_labels[m] + '_Z{0}'.format(inx))
        #     markers_set[marker_labels[m]] = pd.DataFrame(data=m_np, columns=['X{0}'.format(inx), 'Y{0}'.format(inx), 'Z{0}'.format(inx)])
        #     inx += 1
        #
        # trc = TRC(frames_block, col_names=col_names, filename=data[:data.rindex(".")]+".trc", fill_data=fill_data)
        # trc.headers = trc_header
        # trc.marker_set = markers_set
        # trc.marker_names = marker_labels[1:]
        # return trc

    def z_up_to_y_up(self):
        offset = 2
        n = 0
        for m in self.marker_set:
            k = self.marker_set[m]
            j = k.to_numpy()
            r = Rotation.from_euler('xyz', [-90, -90, 0], degrees=True)
            y = np.matmul(r.as_matrix(), j.T)
            start = offset + n
            end = start + 3
            self.data[:, start: end] = y.T
            col = [c for c in k.columns]
            p = pd.DataFrame(data=y.T, columns=col)
            self.marker_set[m] = p
            n += 3
        self.update()

    def x_up_to_y_up(self):
        offset = 2
        n = 0
        for m in self.marker_set:
            k = self.marker_set[m]
            j = k.to_numpy()
            r = Rotation.from_euler('xyz', [-90, 0, 90], degrees=True)
            y = np.matmul(r.as_matrix(), j.T)
            r0 = Rotation.from_euler('xyz', [0, 180, 0], degrees=True)
            y = np.matmul(r0.as_matrix(), y)
            start = offset + n
            end = start + 3
            self.data[:, start: end] = y.T
            col = [c for c in k.columns]
            p = pd.DataFrame(data=y.T, columns=col)
            self.marker_set[m] = p
            n += 3
        self.update()

    def update_from_markerset(self):
        offset = 2
        n = 0
        ms = []
        a = 0
        num_markers = len(self.marker_set.keys())
        temp = np.zeros([self.data.shape[0], 3*num_markers+2])
        temp[:, :self.data.shape[1]] = self.data
        for m in self.marker_set:
            k = self.marker_set[m]
            j = k.to_numpy()
            start = offset + n
            end = start + 3
            temp[:, start: end] = j
            col = ["X{0}".format(a+1), "Y{0}".format(a+1), "Z{0}".format(a+1)]
            p = pd.DataFrame(data=j, columns=col)
            self.marker_set[m] = p
            n += 3
            a += 1
            ms.append(m)
        self.data = temp
        self.headers[MocapFlags.Units.value] = "m"
        self.headers[MocapFlags.NumFrames.value] = 1
        self.headers[MocapFlags.OrigNumFrames.value] = 1
        self.headers[MocapFlags.NumMarkers.value] = len(ms)

        col = [c for c in self.col_labels if ('frame' in c.lower() or 'time' in c.lower()) or c[:c.rindex('_')] in ms]
        self.col_labels = col
        self.marker_names = ms
        self.update()

    def update(self, units=None):
        if len([k for k in self.marker_set.keys()]) > 0:
            backup = deepcopy(self.data)
            marker_temp = {}
            self.x = self.data[:, 1]
            trc_header = {k.value: -1 for k in MocapFlags.defaults_to_list()}
            dt = 0.01
            if not np.isnan(self.dt):
                dt = self.dt
            trc_header[MocapFlags.DataRate.value] = int(1/dt)
            trc_header[MocapFlags.CameraRate.value] = int(1/dt)
            trc_header[MocapFlags.NumFrames.value] = self.data.shape[0]
            if units is None:
                trc_header[MocapFlags.Units.value] = self.headers['Units']
            else:
                trc_header[MocapFlags.Units.value] = units

            trc_header[MocapFlags.OrigDataStartFrame.value] = 1
            trc_header[MocapFlags.OrigDataRate.value] = int(1/dt)
            trc_header[MocapFlags.OrigNumFrames.value] = self.data.shape[0]
            trc_header[MocapFlags.NumMarkers.value] = len(self.marker_names)
            self.headers = trc_header
            for m in self.marker_set:
                for j in range(0, len(self.column_labels)):
                    label = self.column_labels[j]
                    if m in label and "x" in label.lower():
                        st = j
                        en = j + 3
                        mx = deepcopy(self.data[:, st:en])
                        marker_temp[m] = pd.DataFrame(data=mx, columns=self.marker_set[m].columns)
                        #self.marker_set[m].iloc[:, :] = self.data[:, st:en]
                        break

            self.marker_set = marker_temp
            self.data = backup
            pass
        else:
            def split_cols(k):
                idxs = k.split('_')
                if len(idxs) <= 2:
                    idx0 = idxs[0]
                    return idx0
                else:
                    idx0 = ""
                    for j in range(0, len(idxs) - 1):
                        idx0 += "{0}_".format(idxs[j])
                    idx0 = idx0[:-1]
                    return idx0
            col = [c for c in self.column_labels if 'time' not in c]
            kcol = []
            for k in col:
                idx = split_cols(k)
                if idx is not None:
                    kcol.append(idx)

            markers = {kcol[c]: pd.DataFrame(data=np.zeros([self.data.shape[0], 3]), columns=["X", "Y", "Z"]) for c in range(1, len(kcol)) if 'time' not in kcol[c].lower()}
            marker_names = [c for c in markers.keys()]
            if self.data[0, 0] > 1:
                self.data[:, 0] = self.data[:, 0] - self.data[0, 0] +1
            for m in range(0, len(marker_names)):
                markers[marker_names[m]].columns = ['X{0}'.format(m+1), 'Y{0}'.format(m+1), 'Z{0}'.format(m+1)]
            trc_header = {k.value: -1 for k in MocapFlags.defaults_to_list()}
            trc_header[MocapFlags.DataRate.value] = int(1/self.dt)
            trc_header[MocapFlags.CameraRate.value] = int(1/self.dt)
            trc_header[MocapFlags.NumFrames.value] = self.data.shape[0]
            trc_header[MocapFlags.Units.value] = 'm'

            trc_header[MocapFlags.OrigDataStartFrame.value] = 1
            trc_header[MocapFlags.OrigDataRate.value] = int(1/self.dt)
            trc_header[MocapFlags.OrigNumFrames.value] = self.data.shape[0]
            trc_header[MocapFlags.NumMarkers.value] = len(marker_names)
            self.headers = trc_header
            self.marker_names = marker_names
            for c in range(2, len(self.column_labels), 3):
                marker = split_cols(self.column_labels[c])
                markers[marker].iloc[:, :] = self.data[:, c:c+3]
            self.marker_set = markers
            temp = np.zeros([self.data.shape[0], self.data.shape[1]])
            temp[:, 0] = [i + 1 for i in range(0, self.data.shape[0])]
            temp[:, 1:] = self.data[:, 1:]
            self.data = temp
            self.x = self.data[:, 1]
            pass
        if self.data.shape[0] > 3:
            self.update_Spline()

    def write(self, filename):
        lines = [self.first_line + "\n"]
        line = ""
        for s in self.headers_labels:
            line += s.value+"\t"
        lines.append(line.strip()+"\n")
        line = ""
        for s in self.headers_labels:
            line += "{0}\t".format(self.headers[s.value])
        lines.append(line.strip() + "\n")
        c0 = "Frame#\tTime\t"
        c1 = "\t\t"
        for c in self.marker_names:
            c0 += c + "\t\t\t"
            d = [k for k in self.marker_set[c].columns]
            c1 += "{0}\t{1}\t{2}\t".format(d[0], d[1], d[2])
        lines.append(c0.strip() + "\t\t\n")
        lines.append("\t\t" + c1.strip() + "\n")
        for si in range(0, self.data.shape[0]):
            line = str(int(self.data[si, 0])) + '\t' + str(self.data[si, 1])
            for m in self.marker_names:
                marker = self.marker_set[m]
                x = marker.iloc[si, 0]
                y = marker.iloc[si, 1]
                z = marker.iloc[si, 2]

                xo = "{0:.5f}".format(x)
                yo = "{0:.5f}".format(y)
                zo = "{0:.5f}".format(z)
                if np.isnan(x):
                    xo = ''
                if np.isnan(y):
                    yo = ''
                if np.isnan(z):
                    zo = ''
                line += '\t' + str(xo) + '\t' + str(yo) + '\t' + str(zo)
            line += '\n'
            lines.append(line)
        with open(filename, 'w') as writer:
            writer.writelines(lines)
        pass


class MocapFlags(Enum):
    DataRate = 'DataRate'
    CameraRate = 'CameraRate'
    NumFrames = 'NumFrames'
    NumMarkers = 'NumMarkers'
    Units = 'Units'
    OrigDataRate = 'OrigDataRate'
    OrigDataStartFrame = 'OrigDataStartFrame'
    OrigNumFrames = 'OrigNumFrames'
    mm = 'mm'
    m = 'm'

    @staticmethod
    def unit(unit:str):
        if MocapFlags.mm.value == unit:
            return MocapFlags.mm
        else:
            return MocapFlags.m

    @staticmethod
    def defaults_to_list():
        return [k for k in MocapFlags if isinstance(k.value, str) if k.value != 'mm' or k.value != 'm']


class OSIMStorage(Yatsdo):
    """
    Coordinates
    version=1
    nRows=4269
    nColumns=40
    inDegrees=yes

    Units are S.I. units (second, meters, Newtons, ...)
    If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).

    endheader
    """
    def __init__(self, data, col_names=None, fill_data=False, filename="", header=None, ext=".sto"):
        super().__init__(data, col_names, fill_data)
        self.filename = filename
        self.header = header
        self.ext = ext

    @staticmethod
    def yes_no_to_true_false(wo):
        if 'yes' in wo.lower():
            return True
        elif 'no' in wo.lower():
            return False
        return None

    @staticmethod
    def true_false_to_yes_no(wo):
        if wo:
            return 'yes'
        elif not wo:
            return 'no'
        return None

    @staticmethod
    def read(filename):
        if not os.path.exists(filename):
            return None
        s = StorageIO.load(filename, StorageType.mot)

        k = s.info["header"]
        header = {"type": k[0].strip(),
                  "version": int(k[1].split('=')[1].strip()),
                  "nRows": int(k[2].split('=')[1].strip()),
                  "nColumns": int(k[3].split('=')[1].strip()),
                  "inDegrees": OSIMStorage.yes_no_to_true_false(k[4].split('=')[1].strip()),
                  "note": "{0}{1}".format(k[6], k[7]),
                  "end": k[9]
                  }
        ret = OSIMStorage(s.data, filename=filename, header=header)
        return ret

    def update(self):
        super().update()
        self.header['nRows'] = self.data.shape[0]
        self.header['nColumns'] = self.data.shape[1]
        pass

    def header2string(self, h):
        if h not in ['type', 'note', 'end']:
            if h not in ['inDegrees']:
                return "{0}={1}\n".format(h, self.header[h])
            else:
                return "{0}={1}\n".format(h, OSIMStorage.true_false_to_yes_no(self.header[h]))
        else:
            return "{0}{1}\n".format('', self.header[h])

    def write(self, filename):
        lines = [self.header2string(h) for h in self.header]
        cols = ""
        for c in self.col_labels:
            cols += c
            cols += "\t"
        lines.append(cols.strip() + "\n")
        for i in range(0, self.data.shape[0]):
            ret = ""
            for j in range(0, self.data.shape[1]):
                ret += "{0:.8f}\t".format(self.data[i,j])
            ret.strip()
            ret += '\n'
            lines.append(ret)
            pass
        # f = open(filename + ".csv", "r")
        # count = 0
        # stream = "Hello world"
        # while len(stream) > 0:
        #     stream = f.readline()
        #     if count == 0:
        #         count += 1
        #         continue
        #     else:
        #         count += 1
        #         lines.append(stream)
        # f.close()

        with open(filename, 'w') as writer:
            writer.writelines(lines)
        pass
