from opensim.simbody import Vector
import opensim as osm
import pandas as pd
import copy
import numpy as np

from ptb.util.osim.osim_store import OSIMStorage
from ptb.util.io.mocap.file_formats import TRC
from ptb.util.gait.opsim import OsimModel


class OsimHelper:
    def __init__(self, model_path: str, custom_geometry_path=r"C:\OpenSim 4.5\Geometry"):
        """
        This is a helper for osim model gait2392. It may work for other models but not tested.
        :param model_path: model file path
        """
        # set up helper
        osm.ModelVisualizer.addDirToGeometrySearchPaths(custom_geometry_path)
        self.osim_model = OsimModel(model_path)
        cn = self.osim_model.osim.getStateVariableNames()
        self.state_variable_names = [cn.get(c) for c in range(0, cn.getSize())]
        self.state_variable_names_processed = [c.split("/")[-2] if 'value' in c else 'N_A' for c in self.state_variable_names]

        m = self.osim_model.markerset
        self.osim_markers = [f for f in m.marker_set]
        self.__markerset__ = {}
        for x in m.marker_set:
            p = m.marker_set[x].location_in_ground
            self.__markerset__[x] = p

        self.locked = ["mtp_r", "mtp_l"]
        self.locked_id = []
        for f in range(0, len(self.osim_model.joint_names)):
            y = self.osim_model.joint_names[f]
            if y in self.locked:
                self.locked_id.append(1)
            else:
                self.locked_id.append(0)

    @property
    def markerset(self):
        """
        Access function to get marker set
        :return: current marker position of the marker set (dict)
        """
        return pd.DataFrame(self.__markerset__)

    def update(self):
        """
        Update the helper with the current marker position from the model
        :return:
        """
        m = self.osim_model.markerset
        for x in m.marker_set:
            self.__markerset__[x] = copy.deepcopy(m.marker_set[x].location_in_ground)
        pass

    def set_joint(self, joint_name, v):
        j = None
        if isinstance(joint_name, int):
            j = self.osim_model.jointset[self.osim_model.joint_names[joint_name]]
        elif isinstance(joint_name, str):
            j = self.osim_model.jointset[joint_name]
        if j is not None:
            b = j.set_joint(v)
            if b == 0:
                self.update()
            elif b == -1:
                print("Can not set {0} due to num of coord is not the same: v({1}) != {0}({2})".format(joint_name, len(v), j.num_coord))
        else:
            print("Can not set {0} due to it not being in the model.".format(joint_name))

    def set_joints(self, v):
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
            ret = [0 for n in range(0, len(self.state_variable_names))]
            cols = None
            if isinstance(v, pd.DataFrame):
                cols = [c for c in v.columns if 'time' not in c]
            if isinstance(v, pd.Series):
                cols = [c for c in v.index if 'time' not in c]
            kl = {c: self.state_variable_names_processed.index(c) for c in cols}
            for c in cols:
                vx = 0
                if isinstance(v, pd.DataFrame):
                    vx = float(v[c].iloc[0])
                    if not ('tx' in c or 'ty' in c or 'tz' in v[c]):
                        vx = np.pi * (vx / 180.0)
                if isinstance(v, pd.Series):
                    vx = float(v[c])
                    if not ('tx' in c or 'ty' in c or 'tz' in c):
                        vx = np.pi * (vx / 180.0)

                ret[kl[c]] = vx
            vx1 = Vector(ret)
            self.osim_model.osim.setStateVariableValues(self.osim_model.state, vx1)
            self.osim_model.osim.assemble(self.osim_model.state)
            self.update()

    def export_marker_data_from_motion(self, mot_path):
        """
        This is a helper method exports model markers to TRC based on motion file (mot or sto)
        :param mot_path: model file path
        """
        k0 = copy.deepcopy(self.markerset)
        idx = 1
        index_m = {}
        columns = ['Frame#', 'Time']
        for c in k0.columns:
            index_m[c] = idx
            columns.append("{1}_X{0}".format(index_m[c], c))
            columns.append("{1}_Y{0}".format(index_m[c], c))
            columns.append("{1}_Z{0}".format(index_m[c], c))
            idx += 1

        m = OSIMStorage.read(mot_path)
        frame_rate = int(1 / m.store.dt)

        frames = []
        for i in range(0, m.store.data.shape[0]):
            joint = pd.Series(data=m.store.data[i, :], index=m.store.column_labels)
            self.set_joints(joint)
            frames.append(copy.deepcopy(self.markerset))

        marker_df = np.zeros([len(frames), len(columns)])
        for i in range(0, len(frames)):
            marker_df[i, 0] = i + 1
            marker_df[i, 1] = i * 1.0 / frame_rate
            frame = frames[i]
            for j in range(0, frame.shape[1]):
                st = j * 3 + 2
                en = st + 3
                marker_df[i, st: en] = frame.iloc[:, j].to_numpy()
                pass
            pass

        df = pd.DataFrame(data=marker_df, columns=columns)
        trc = TRC(df)
        trc.headers['DataRate'] = frame_rate
        trc.headers['CameraRate'] = frame_rate
        trc.headers['OrigDataRate'] = frame_rate
        trc.headers['NumFrames'] = len(frames)
        trc.headers['OrigNumFrames'] = len(frames)
        trc.update()
        out_file = "{0}.trc".format(mot_path[:mot_path.rindex('.')])
        trc.write(out_file)