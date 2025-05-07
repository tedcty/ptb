from ptb.util.gait.opsim import OsimModel
from opensim.simbody import Vector
import pandas as pd
import copy
import numpy as np


class OsimHelper:
    def __init__(self, model_path: str):
        """
        This is a helper for osim model gait2392. It may work for other models but not tested.
        :param model_path: model file path
        """
        # set up helper
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