# coding=utf-8
import os
from enum import Enum
import xml.sax
from xml.dom.minidom import parse, Node
import xml.dom.minidom

import matplotlib.pyplot as plt
import numpy as np
import csv
import opensim as om
import subprocess
from opensim import InverseKinematicsTool
from ptb.util.io.helper import StorageIO, StorageType, MYXML
from ptb.util.io.mocap.file_formats import TRC
from ptb.util.math.filters import Butterworth
from ptb.util.math.transformation import Cloud
from scipy import interpolate
from scipy.signal import find_peaks


class OsimModel:
    def __init__(self, filepath):
        self.osim = om.Model(filepath)
        self.osim.initSystem()
        self.osim.initializeState()
        self.state = self.osim.getWorkingState()
        self.body_names = [self.osim.get_BodySet().get(n).getName() for n in range(0, self.osim.getNumBodies())]
        self.bodyset = {self.body_names[n]: Body(self.body_names[n], self) for n in range(0, len(self.body_names))}
        self.markerset = MarkerSet(self)
        self.joint_names = [self.osim.get_JointSet().get(n).getName() for n in range(0, self.osim.getNumJoints())]
        self.jointset = {self.joint_names[n]: Joint(self.joint_names[n], self) for n in range(0, len(self.joint_names))}

    def re_init(self):
        # self.osim.initSystem()
        self.osim.calcMassCenterPosition(self.state)
        self.osim.calcMassCenterVelocity(self.state)
        self.osim.calcMassCenterAcceleration(self.state)
        # self.osim.initializeState()
        # self.state = self.osim.getWorkingState()

    def get_body(self, n):
        return self.osim.get_BodySet().get(n)

    def get_joint(self, n):
        return self.osim.get_JointSet().get(n)

    def get_marker_set(self):
        return self.osim.get_MarkerSet()

    def ground(self):
        return self.osim.getGround()


class Body:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.__body__: om.Body = self.model.get_body(self.model.body_names.index(name))
        self.markers = {}

    @property
    def obody(self):
        return self.__body__

    def express_vector_in_ground(self, state, loc):
        return OsimTranslator.vec3(self.__body__.expressVectorInGround(state, loc))

    def get_position_in_ground(self, state):
        return self.__body__.getPositionInGround(state)


class Joint:
    def __init__(self, name, model):
        self.model = model
        self.name = name
        self.__joint__ = self.model.get_joint(self.model.joint_names.index(name))
        self.num_coord = self.__joint__.numCoordinates()
        # print("num coord - {0}: {1}".format(self.name, self.num_coord))
        temp = [[self.__joint__.get_coordinates(i), i] for i in range(0, self.num_coord)]
        self.coord = {c[0].getName(): c for c in temp}
        self.parent_offset_ground = OsimTranslator.vec3(
            self.__joint__.getParentFrame().getPositionInGround(self.model.state))

    def update(self):
        self.parent_offset_ground = OsimTranslator.vec3(
            self.__joint__.getParentFrame().getPositionInGround(self.model.state))

    def get_coordinate(self, i):
        c = None
        if isinstance(i, int):
            c = self.__joint__.get_coordinates(i)
            return c
        if isinstance(i, str):
            try:
                c = self.coord[i]
            except KeyError:
                print("coordinate {0} does not exist")
        return c

    def set_joint(self, v, coord=0, f=False):
        if isinstance(v, int):
            if coord < self.num_coord:
                c0 = self.__joint__.upd_coordinates(coord)
                c0.setValue(self.model.state, v)
                return 0
        if isinstance(v, list):
            if len(v) == self.num_coord:
                for i in range(0, len(v)):
                    c0 = self.__joint__.upd_coordinates(i)
                    c0.setValue(self.model.state, v[i], f)
                return 0
        if isinstance(v, np.ndarray):
            if v.shape[0] == self.num_coord:
                for i in range(0, v.shape[0]):
                    c0 = self.__joint__.upd_coordinates(i)
                    c0.setValue(self.model.state, v[i])
                return 0
        return -1


class MarkerSet:
    def __init__(self, osim):
        self.model = osim
        self.__marker_set__ = self.model.get_marker_set()
        markernames = om.ArrayStr()
        self.__marker_set__.getMarkerNames(markernames)
        self.marker_names = [markernames.get(m) for m in range(0, markernames.size())]
        self.marker_set = {n: Marker(n, self.get(n), osim) for n in self.marker_names}

    def get(self, name):
        return self.__marker_set__.get(name)


class OsimTranslator:
    @staticmethod
    def vec3(v):
        return np.array([v[0], v[1], v[2]])


class Marker:
    def __init__(self, name, marker, model):
        self.model = model
        self.marker_name = name
        self.__marker__ = marker
        self.location = OsimTranslator.vec3(marker.get_location())
        self.__location__ = marker.get_location()
        self.body_path = marker.getParentFrameName()

        self.body_name = ""
        for b in self.model.body_names:
            if b in self.body_path:
                self.body_name = b
                break
        self.body: Body = self.model.bodyset[self.body_name]
        self.body.markers[self.marker_name] = self

        self.debug = False
        pass

    @property
    def omarker(self):
        return self.__marker__

    @property
    def location_in_ground(self):
        self.location = OsimTranslator.vec3(self.__marker__.get_location())
        lo = np.atleast_2d([self.location[0], self.location[1], self.location[2], 1])
        p = self.body.get_position_in_ground(self.model.osim.getWorkingState())
        tg = self.body.obody.getTransformInGround(self.model.state)
        r33 = tg.R()
        t13 = tg.T()
        mat44np = np.eye(4, 4)
        mat44np[:3, :3] = np.array([[r33.get(m, n) for n in range(0, 3)] for m in range(0, 3)])
        mat44np[:3, 3] = np.array([t13.get(m) for m in range(0, 3)])
        nlo = np.matmul(mat44np, lo.T)
        # p = self.body.express_vector_in_ground(self.model.osim.getWorkingState(), lo)
        if self.debug:
            print("\t{0}, {1}, {2}".format(p[0], p[1], p[2]))
            print("+\t{0}, {1}, {2}".format(lo[0], lo[1], lo[2]))
        return np.squeeze(nlo[:3, 0])


class ID:
    @staticmethod
    def run(xml_file):
        subprocess.run(["id", "-S", xml_file])


class IK:
    @staticmethod
    def run(xml_file):
        subprocess.run(["ik", "-S", xml_file])

    @staticmethod
    def run_from_c3d(wkdir="M:/test/", root="M:/Mocap/P011/New Session/",
                     trial_c3d="Straight normal 1.c3d",
                     template='M:/template/Straight normal 1.xml',
                     mode=0):
        if mode == 0:
            IK.run_from_c3d_0(wkdir, trial_c3d, template)
        elif mode == 1:
            IK.run_from_c3d_1(wkdir, trial_c3d, template)

    @staticmethod
    def run_from_c3d_0(wkdir="M:/test/",
                       trial_c3d="Straight normal 1.c3d",
                       template='M:/template/Straight normal 1.xml'):

        # read task
        trc_name = trial_c3d[:trial_c3d.rindex('.c3d')]
        trc = TRC.create_from_c3d(trial_c3d)
        trc.z_up_to_y_up()
        trc.write("{0}{1}.trc".format(wkdir, trc_name))
        ik = InverseKinematicsTool(template)
        b = ik.get_output_motion_file()
        bf = "{0}/{1}.sto".format(b[:b.rindex("/")], trc_name)
        ik.set_output_motion_file(bf)

        x = trc.data[:, 1]
        ik.setStartTime(x[0])
        ik.setEndTime(x[-1])

        try:
            ik.run()
        except RuntimeError:
            pass
        pass

    @staticmethod
    def run_from_c3d_1(wkdir="M:/test/",
                       trial_c3d="Straight normal 1.c3d",
                       template='M:/template/Straight normal 1.xml',
                       model=""):

        # read task
        trc_name = trial_c3d[:trial_c3d.rindex('.c3d')]
        c3d_file = "{0}{1}".format(wkdir, trial_c3d)
        trc = TRC.create_from_c3d(c3d_file)
        trc.z_up_to_y_up()
        # trc.fill_gaps()
        segments = {'l_forearm': ["L_Med_HumEpicondyle", "L_Lat_HumEpicondyle", "L_Ulna", "L_Radius"],
                    'l_upperarm': ["L_Med_HumEpicondyle", "L_Lat_HumEpicondyle", 'L_Acromion'],
                    'r_forearm': ["R_Med_HumEpicondyle", "R_Lat_HumEpicondyle", "R_Ulna", "R_Radius"],
                    'r_upperarm': ["R_Med_HumEpicondyle", "R_Lat_HumEpicondyle", 'R_Acromion'],
                    'torso': ['Sternum', 'R_Acromion', 'L_Acromion'],
                    'pelvis': ['R_ASIS', 'L_ASIS', 'R_PSIS', 'L_PSIS'],
                    'l_shank': ['L_LatKnee', 'L_MedKnee', 'L_FibHead', 'L_MidShank', 'L_LatAnkle', 'L_MedAnkle'],
                    'r_shank': ['R_LatKnee', 'R_MedKnee', 'R_FibHead', 'R_MidShank', 'R_LatAnkle', 'R_MedAnkle'],
                    'l_foot': ['L_LatAnkle', 'L_MedAnkle', "L_DP1", "L_MT2", "L_MT5", "L_Heel"],
                    'r_foot': ['R_LatAnkle', 'R_MedAnkle', "R_DP1", "R_MT2", "R_MT5", "R_Heel"]
                    }

        segments_ref = {}
        count = 0
        for i in range(0, trc.data.shape[0]):
            for s in segments:
                try:
                    a = segments_ref[s]
                except KeyError:
                    if s == 'r_forearm':
                        pass
                    matr0 = np.zeros([3, len(segments[s])])
                    boo = False
                    for m in range(0, len(segments[s])):
                        c = trc.marker_set[segments[s][m]].iloc[i, :].to_list()
                        if np.sum(np.isnan(c)) > 0:
                            boo = False
                            break
                        matr0[:, m] = c
                        boo = True
                    if boo:
                        segments_ref[s] = matr0
        segments_ref_dis = {}
        for s in segments_ref:
            pos = []
            seg_dis = {}
            for m in segments_ref[s]:
                pos.append(m)
            pos_mat = (np.asarray(pos)).T
            avg = np.mean(pos_mat, axis=0)
            seg_dis['centre'] = avg

            for m in range(0, len(segments[s])):
                seg_dis[segments[s][m]] = np.linalg.norm(segments_ref[s][:, m] - avg)
            segments_ref_dis[s] = seg_dis
            pass

        marker_acc = {}
        marker_sig = {}
        marker_sig_peaks = {}
        for m in trc.marker_set:
            print(m)
            marker = trc.marker_set[m].to_numpy()
            marker_dt = marker[1:, :] - marker[:-1]
            marker_dt = np.array([np.linalg.norm(marker_dt[i, :]) for i in range(0, marker_dt.shape[0])])
            marker_dtp = marker_dt[1:] - marker_dt[:-1]
            acc = np.squeeze(np.zeros([1, marker.shape[0]]))
            acc[2:] = marker_dtp

            accsq = np.array([acc[i] * acc[i] for i in range(0, len(acc))])
            meana = np.sqrt(np.nanmean(accsq))
            med = np.sqrt(np.nanmedian(accsq))
            stda = np.nanstd(accsq)
            mx = np.nanmax(accsq)
            u = meana / stda
            v = (mx - (3 * stda)) / u
            score = np.abs(v) / 1e5
            print(np.abs(v) / 1e5)
            if score > 1:
                ar, _ = find_peaks(accsq, height=mx * 0.8)
                marker_sig_peaks[m] = ar
                marker_acc[m] = acc
            # plt.figure()
            # x = np.array([i for i in range(0, acc.shape[0])])
            # plt.title(m)
            # plt.plot(acc)
            # plt.plot(x[ar], acc[ar], 'x')
            # plt.show()

        def get_segment(m):
            for j in segments:
                if m in segments[j]:
                    return j

        unlabelled = [m for m in trc.marker_set if m.startswith('*')]
        for m in marker_sig_peaks:
            print(m)
            c = marker_sig_peaks[m]
            seg = get_segment(m)
            for f in c:
                frame = f
                possibles = []
                for n in marker_sig_peaks:
                    d = marker_sig_peaks[n]
                    for o in d:
                        if o == frame:
                            possibles.append(n)
                            break
                pos = (trc.marker_set[m].to_numpy())[frame, :]
                pos_prev = (trc.marker_set[m].to_numpy())[frame - 1, :]
                possib = {}
                possib_diff = {}
                b = ""
                bb = 100000
                for p in possibles:
                    pos_dt = (trc.marker_set[p].to_numpy())[frame, :]
                    possib[p] = pos_dt
                    possib_diff[p] = np.linalg.norm(pos_dt - pos_prev)
                    if bb > possib_diff[p]:
                        b = p
                        bb = possib_diff[p]
                    pass
                unlabelled_possib_diff = {}
                ubb = 100000
                ub = ""
                unlabelled_possib = {}
                for u in unlabelled:
                    pos_dt = (trc.marker_set[u].to_numpy())[frame, :]
                    unlabelled_possib[u] = pos_dt
                    if np.sum(np.isnan(pos_dt)) > 0:
                        continue
                    unlabelled_possib_diff[u] = np.linalg.norm(pos_dt - pos_prev)
                    if ubb > unlabelled_possib_diff[u]:
                        ub = u
                        ubb = unlabelled_possib_diff[u]
                if ubb > bb:
                    if bb < 30:
                        trc.marker_set[b].iloc[frame, :] = pos
                        trc.marker_set[m].iloc[frame, :] = possib[b]
                else:
                    if ubb < 30:
                        trc.marker_set[ub].iloc[frame, :] = pos
                        trc.marker_set[m].iloc[frame, :] = unlabelled_possib[ub]

                pass

            pass
        segments_ref = {}
        count = 0
        for i in range(0, trc.data.shape[0]):
            for s in segments:
                try:
                    a = segments_ref[s]
                except KeyError:
                    if s == 'r_forearm':
                        pass
                    matr0 = np.zeros([3, len(segments[s])])
                    boo = False
                    for m in range(0, len(segments[s])):
                        c = trc.marker_set[segments[s][m]].iloc[i, :].to_list()
                        if np.sum(np.isnan(c)) > 0:
                            boo = False
                            break
                        matr0[:, m] = c
                        boo = True
                    if boo:
                        segments_ref[s] = matr0

        # for i in range(0, trc.data.shape[0]):
        #     for s in segments:
        #         matr0 = segments_ref[s]
        #         matr1 = []
        #         matr2 = []
        #         indexs = []
        #         indexs_ab = []
        #         for m in range(0, len(segments[s])):
        #             c = trc.marker_set[segments[s][m]].iloc[i, :].to_list()
        #             if np.sum(np.isnan(c)) > 0:
        #                 indexs_ab.append(m)
        #                 continue
        #             matr1.append(matr0[:, m])
        #             matr2.append(c)
        #             indexs.append(m)
        #             pass
        #
        #
        #         if len(matr2) != len(segments[s]):
        #             if len(matr2) >= 3:
        #                 matr_np = (np.array(matr1)).T
        #                 matr2_np = (np.array(matr2)).T
        #                 matr_ones = np.ones([4, len(segments[s])])
        #                 matr_ones[:3, :] = matr0
        #                 t = Cloud.rigid_body_transform(matr_np, matr2_np)
        #                 test = np.matmul(t, matr_ones)
        #                 matr3 = test[:3, :]
        #                 for h in range(0, len(indexs)):
        #                     matr3[:, indexs[h]] = matr2_np[:, h]
        #                 for m in range(0, len(segments[s])):
        #                     k = np.nanmean([matr3[:, m], trc.marker_set[segments[s][m]].iloc[i, :]], axis=0)
        #                     trc.marker_set[segments[s][m]].iloc[i, :] = k
        #                 pass
        #         pass
        #
        #
        #     pass
        cols = [c for c in range(0, len(trc.column_labels)) if '*' not in trc.column_labels[c]]
        data = trc.data[:, cols]
        trc.data = data
        # for m in trc.marker_set:
        #     marker = trc.marker_set[m].to_numpy()
        #     marker_dt0 = []
        #     for i in range(0, 3):
        #         marker_dt0.append(Butterworth.butter_low_filter(marker[:, i], 6, 100))
        #     r = (np.array(marker_dt0)).T
        #     trc.marker_set[m].iloc[:, :] = r
        #
        trc.update_from_markerset()
        trc.headers['Units'] = 'mm'
        trial = "{0}{1}_ex.trc".format(wkdir, trc_name)
        trc.write(trial)
        output_motion_file = "{0}{1}.sto".format(wkdir, trc_name)
        save_name = "{0}{1}_ik_setup.xml".format(wkdir, trc_name)
        IK.write_ik_setup(trial, template, model, output_motion_file, save_name)
        os.listdir()
        ik = InverseKinematicsTool(save_name)
        b = ik.get_output_motion_file()
        bf = "{0}/{1}.sto".format(b[:b.rindex("/")], trc_name)
        ik.set_output_motion_file(bf)

        x = trc.data[:, 1]
        ik.setStartTime(x[0])
        ik.setEndTime(x[-1])

        try:
            ik.run()
        except RuntimeError:
            pass
        pass

    @staticmethod
    def write_ik_setup(trial, template, model, output_motion_file, save_name):
        """
        :param trial:
        :param template:
        :param model:
        :param output_motion_file:
        :param save_name:
        """

        trc: TRC = StorageIO.load(trial, StorageType.trc)
        m = MYXML(filename=template)
        pretty = True
        try:
            m.set_value("model_file", model)
        except IndexError:
            m.add_node("InverseKinematicsTool", "model_file", model)
        try:
            m.set_value("output_motion_file", output_motion_file)
        except IndexError:
            m.add_value("output_motion_file", output_motion_file)
        try:
            m.set_value("marker_file", trial)
        except IndexError:
            m.add_value("marker_file", trial)
        try:
            m.set_value("time_range", " " + str(trc.data[0, 1]) + " " + str(trc.data[-1, 1]))
        except IndexError:
            m.add_value("time_range", " " + str(trc.data[0, 1]) + " " + str(trc.data[-1, 1]))
        m.write(save_name, pretty)

    @staticmethod
    def write_ik_setup_xml(ikconfig, save_name):
        IK.write_ik_setup(ikconfig["marker_file"],
                          ikconfig["template"],
                          ikconfig["model_file"],
                          ikconfig["output_motion_file"],
                          save_name)


class Para(Enum):
    force_label = 0
    cutoff = 1
    sampling_rate = 2
    order = 3


class General:
    @staticmethod
    def filer_force_data(s: StorageIO, para: dict = None):
        # default values
        forces = ["ground_force1_vx", "ground_force1_vy", "ground_force1_vz", "ground_force2_vx", "ground_force2_vy",
                  "ground_force2_vz", "ground_force3_vx", "ground_force3_vy", "ground_force3_vz"]
        cut_off = 6
        sampling_rate = 1000
        order = 4
        if para is not None:
            try:
                forces = para[Para.force_label]
            except KeyError:
                pass
            try:
                cut_off = para[Para.cutoff]
            except KeyError:
                pass
            try:
                sampling_rate = para[Para.sampling_rate]
            except KeyError:
                pass
            try:
                order = para[Para.order]
            except KeyError:
                pass
        force_data = s.data[forces]
        for d in range(1, force_data.shape[1]):
            data = s.data.iloc[:, d]
            filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=order)
            force_data.iloc[:, d] = filtered
        for c in forces:  # column headings
            s.data[c] = force_data[c]
        return s

    @staticmethod
    def write_mot(s: StorageIO, filename: str):
        lines = []
        for h in s.info['header']:
            lines.append(h.strip() + "\n")
        cols = ""
        for c in s.data.columns:
            cols += c
            cols += "\t"
        lines.append(cols.strip() + "\n")
        s.data.to_csv(filename + "_temp.csv", sep="\t", index=False)
        f = open(filename + "_temp.csv", "r")
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

