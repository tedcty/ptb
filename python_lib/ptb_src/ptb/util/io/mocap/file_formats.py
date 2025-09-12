import copy
import time
from copy import deepcopy
from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ptb.core import Yatsdo
from ptb.util.io.mocap.low_lvl import c3d

v = np.__version__[0]
if int(v) >= 2:
    nan = np.nan
else:
    nan = np.NaN



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
                    dd.append(nan)
            while len(dd) < cols:
                dd.append(nan)
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

    def get_samples_as_trc(self, time_points, filter="*"):
        subset = self.get_samples(time_points, assume_time_first_col=False, as_pandas=True)
        trc = copy.deepcopy(self)
        trc.data = subset
        trc.x = subset[:, 1]
        trc.update()
        return trc

    def get_data_between_timepoints(self, starttime, endtime, freq):
        period = endtime - starttime
        dt = 1.0 / (1.0 * freq)
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
            try:
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
            except ValueError:
                print(marker_labels[m])
                pass
            inx += 1

        trc = TRC(frames_block, col_names=col_names, filename=filename[:filename.rindex(".")] + ".trc",
                  fill_data=fill_data)
        trc.headers = trc_header
        trc.marker_set = markers_set
        trc.marker_names = marker_labels[1:]
        return trc

    @staticmethod
    def __simple_c3d_reader__(filename, exportas_text=False):
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
                            p = np.asarray([nan, nan, nan])
                            break
                    markers[marker_labels[j]].append(p)
                if exportas_text:
                    ret += 'frame {}: point {}, analog {}'.format(
                        i, points, analog)
            box = {'markers': markers,
                   'mocap': {"rates": reader.point_rate, "units": units, "header": reader.header}, 'text': ret,
                   "analog": analogs}
        return box

    @staticmethod
    def create_from_c3d(data, fill_data=False):
        c3d_data = TRC.__simple_c3d_reader__(data)
        return TRC.create_from_c3d_dict(c3d_data, data, fill_data)

    def z_up_to_y_up(self):
        # offset = 2
        # n = 0
        r = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True)

        k = len(self.marker_set.keys())
        d0 = self.data[:, 2:(k*3)+2]
        #j = int((d0.shape[0]*d0.shape[1])/3)
        kp = np.vstack([self.marker_set[m].to_numpy() for m in self.marker_set])
        y = (np.matmul(r.as_matrix(), kp.T)).T

        yp = np.hstack([y[i:i+d0.shape[0], :] for i in range(0, y.shape[0], d0.shape[0])])
        self.data[:, 2:(k * 3) + 2] = yp
        # r1 = Rotation.from_euler('xyz', [0, -90, 0], degrees=True)
        # r = Rotation.from_matrix(np.matmul(r1.as_matrix(), r0.as_matrix()))
        # for m in self.marker_set:
        #     k = self.marker_set[m]
        #     j = k.to_numpy()
        #
        #     y = np.matmul(r.as_matrix(), j.T)
        #     start = offset + n
        #     end = start + 3
        #     self.data[:, start: end] = y.T
        #     col = [c for c in k.columns]
        #     p = pd.DataFrame(data=y.T, columns=col)
        #     self.marker_set[m] = p
        #     n += 3
        self.update()
        return r.as_matrix()

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

    def update_from_markerset(self, reset_units=False):
        offset = 2
        n = 0
        ms = []
        a = 0
        num_markers = len(self.marker_set.keys())
        temp = np.zeros([self.data.shape[0], 3 * num_markers + 2])
        temp[:, :2] = self.data[:, :2]
        temp_cols = [c for c in self.col_labels[:2]]
        for m in self.marker_set:
            k = self.marker_set[m]
            j = k.to_numpy()
            start = offset + n
            end = start + 3
            temp[:, start: end] = j
            col = ["X{0}".format(a + 1), "Y{0}".format(a + 1), "Z{0}".format(a + 1)]
            p = pd.DataFrame(data=j, columns=col)
            self.marker_set[m] = p
            temp_cols.append("{1}_X{0}".format(a + 1, m))
            temp_cols.append("{1}_Y{0}".format(a + 1, m))
            temp_cols.append("{1}_Z{0}".format(a + 1, m))
            n += 3
            a += 1
            ms.append(m)
        self.data = temp
        self.col_labels = temp_cols
        if reset_units:
            self.headers[MocapFlags.Units.value] = "m"
        self.headers[MocapFlags.NumFrames.value] = 1
        self.headers[MocapFlags.OrigNumFrames.value] = 1
        self.headers[MocapFlags.NumMarkers.value] = len(ms)

        # col = [c for c in self.col_labels if ('frame' in c.lower() or 'time' in c.lower()) or c[:c.rindex('_')] in ms]
        # self.col_labels = col
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
            trc_header[MocapFlags.DataRate.value] = int(1 / dt)
            trc_header[MocapFlags.CameraRate.value] = int(1 / dt)
            trc_header[MocapFlags.NumFrames.value] = self.data.shape[0]
            if units is None:
                trc_header[MocapFlags.Units.value] = self.headers['Units']
            else:
                trc_header[MocapFlags.Units.value] = units

            trc_header[MocapFlags.OrigDataStartFrame.value] = 1
            trc_header[MocapFlags.OrigDataRate.value] = int(1 / dt)
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
                        # self.marker_set[m].iloc[:, :] = self.data[:, st:en]
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

            markers = {kcol[c]: pd.DataFrame(data=np.zeros([self.data.shape[0], 3]), columns=["X", "Y", "Z"]) for c in
                       range(1, len(kcol)) if 'time' not in kcol[c].lower()}
            marker_names = [c for c in markers.keys()]
            if self.data[0, 0] > 1:
                self.data[:, 0] = self.data[:, 0] - self.data[0, 0] + 1
            for m in range(0, len(marker_names)):
                markers[marker_names[m]].columns = ['X{0}'.format(m + 1), 'Y{0}'.format(m + 1), 'Z{0}'.format(m + 1)]
            trc_header = {k.value: -1 for k in MocapFlags.defaults_to_list()}
            trc_header[MocapFlags.DataRate.value] = int(1 / self.dt)
            trc_header[MocapFlags.CameraRate.value] = int(1 / self.dt)
            trc_header[MocapFlags.NumFrames.value] = self.data.shape[0]
            trc_header[MocapFlags.Units.value] = 'm'

            trc_header[MocapFlags.OrigDataStartFrame.value] = 1
            trc_header[MocapFlags.OrigDataRate.value] = int(1 / self.dt)
            trc_header[MocapFlags.OrigNumFrames.value] = self.data.shape[0]
            trc_header[MocapFlags.NumMarkers.value] = len(marker_names)
            self.headers = trc_header
            self.marker_names = marker_names
            for c in range(2, len(self.column_labels), 3):
                marker = split_cols(self.column_labels[c])
                markers[marker].iloc[:, :] = self.data[:, c:c + 3]
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
            line += s.value + "\t"
        lines.append(line.strip() + "\n")
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
    def unit(unit: str):
        if MocapFlags.mm.value == unit:
            return MocapFlags.mm
        else:
            return MocapFlags.m

    @staticmethod
    def defaults_to_list():
        return [k for k in MocapFlags if isinstance(k.value, str) if k.value != 'mm' or k.value != 'm']


class ForcePlate(Yatsdo):
    def __init__(self, data, col_names=None):
        super().__init__(data, col_names)
        self.corners = None
        self.origin_offset = None
        self.num_of_plates = -1
        self.plate = {}
        self.units = None

    def rotate(self, r, flip_z=True):
        start = time.time()
        if flip_z:
            r0 = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
            self.__rotate__(r0.as_matrix())
        self.__rotate__(r)
        # Hard coded correction
        for p in self.plate:
            x = self.plate[p]
            xf = x[1]
            # force_labels = [f for f in xf.columns if "Force" in f]
            # force = xf[force_labels].to_numpy()
            # xf[force_labels[0]] = force[:, 2]
            # xf[force_labels[2]] = -1 * force[:, 0]
            #
            # moment_labels = [f for f in xf.columns if "Moment" in f]
            # moment = xf[moment_labels].to_numpy()
            # xf[moment_labels[0]] = moment[:, 2]
            # xf[moment_labels[2]] = moment[:, 0]

            COP_labels = [f for f in xf.columns if "COP" in f]
            COP = xf[COP_labels].to_numpy()
            # xf[COP_labels[0]] = COP[:, 2]
            xf[COP_labels[2]] = -COP[:, 2]
            self.plate[p][1] = xf
        print("force rotation {0}".format(time.time()-start))
        pass

    def __rotate__(self, r, include_cop=True):
        for p in self.plate:
            x = self.plate[p]
            xf = x[1]
            force_labels = [f for f in xf.columns if "Force" in f]
            moment_labels = [f for f in xf.columns if "Moment" in f]
            force = xf[force_labels]
            force_df = (r @ force.to_numpy().T).T
            moment = xf[moment_labels]
            moment_df = (r @ moment.to_numpy().T).T
            if include_cop:
                COP_labels = [f for f in xf.columns if "COP" in f]
                COP = xf[COP_labels]
                COP_df = (r @ COP.to_numpy().T).T
                xf[COP_labels] = COP_df
            xf[force_labels] = force_df
            xf[moment_labels] = moment_df

            self.plate[p][1] = xf
            pass
        pass

    @staticmethod
    def create(param, data):
        f = ForcePlate(data)
        f.corners = param["corners"]
        f.origin_offset = param["origin"]
        f.num_of_plates = param["num_plates"]
        f.sort_plates()
        f.cop()
        return f

    def __moments__(self):
        for p in self.plate:
            xf = self.plate[p][1]
            force_labels = [f for f in xf.columns if "Force" in f]
            force_df = xf[force_labels]
            COP_labels = [f for f in xf.columns if "COP" in f]
            COP_df = xf[COP_labels]
            mo = np.zeros([COP_df.shape[0], 3])
            for i in range(0, COP_df.shape[0]):
                mo[i, 0] = COP_df.iloc[i, 1] * force_df.iloc[i, 2] - COP_df.iloc[i, 2] * force_df.iloc[i, 1]
                mo[i, 1] = COP_df.iloc[i, 2] * force_df.iloc[i, 0] - COP_df.iloc[i, 0] * force_df.iloc[i, 2]
                mo[i, 2] = COP_df.iloc[i, 0] * force_df.iloc[i, 1] - COP_df.iloc[i, 1] * force_df.iloc[i, 0]
            moment_labels = [f for f in xf.columns if "Moment" in f]
            pass

    def sort_plates(self):
        mapper = pd.DataFrame(data=self.data, columns=self.col_labels)
        for i in range(1, self.num_of_plates + 1):
            cols = self.col_labels[:3]
            cols.append("Force.Fx{0}".format(i))
            cols.append("Force.Fy{0}".format(i))
            cols.append("Force.Fz{0}".format(i))
            cols.append("Moment.Mx{0}".format(i))
            cols.append("Moment.My{0}".format(i))
            cols.append("Moment.Mz{0}".format(i))
            self.plate["force_plate_{0}".format(i)] = [i, mapper[cols]]
        pass

    def cop(self, px=None):
        for p in self.plate:
            if px is not None:
                if p != px:
                    continue
            data = self.plate[p][1]
            idx = self.plate[p][0]

            data_col = [c for c in data.columns]
            data_col.append("COP.Px{0}".format(idx))
            data_col.append("COP.Py{0}".format(idx))
            data_col.append("COP.Pz{0}".format(idx))
            data_w_cop = np.zeros([data.shape[0], data.shape[1] + 3])
            data_w_cop[:, :-3] = data
            cop = np.zeros([self.data.shape[0], 3])
            My = data["Moment.My{0}".format(idx)].to_numpy()
            Mx = data["Moment.Mx{0}".format(idx)].to_numpy()
            Fz = data["Force.Fz{0}".format(idx)].to_numpy()
            for i in range(0, self.data.shape[0]):
                ax = -My[i] / Fz[i]
                ay = -Mx[i] / Fz[i]
                cop[i, :] = np.array([ax, ay, 0]) + self.origin_offset[idx - 1]
                pass
            data_w_cop[:, data.shape[1]:] = cop
            self.plate[p] = [idx, pd.DataFrame(data=data_w_cop, columns=data_col)]
            pass

