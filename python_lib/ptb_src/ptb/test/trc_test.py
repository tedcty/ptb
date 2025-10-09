import copy

from ptb.util.io.mocap.file_formats import TRC
from ptb.util.gait.helpers import OsimHelper
from ptb.util.math.transformation import Cloud
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation

class Param:
    # Mapping to Rajagopal model
    xsens_joints = {
        "Marker": ["LACR", "RACR", "LSJC", "RSJC", "LEJC", "REJC", "Sacrum", "T8", "LHJC", "RHJC", "LKJC", "RKJC", "LAJC", "RAJC"],
        "IMU": ["LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm", "LeftForeArm", "RightForeArm", "Pelvis", "T8", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot"]
    }

    # this gives the effect of the IMU
    # xsens_markers = {
    #     "LeftForeArm": ["LEJC", "LFAradius", "LFAulna"],
    #     "RightForeArm": ["REJC", "RFAradius", "RFAulna"],
    #     "LeftUpperArm": ["LSJC", 'LUA1', 'LUA2', 'LUA3', "LEJC", "LFAradius", "LFAulna"],
    #     "RightUpperArm": ["RSJC", 'RUA1', 'RUA2', 'RUA3', "REJC", "RFAradius", "RFAulna"],
    #
    #     # "LeftShoulder": ["LACR", "LSJC",'LUA1', 'LUA2', 'LUA3', "LEJC", "LFAradius", "LFAulna"],
    #     # "RightShoulder": ["RACR", "RSJC",'RUA1', 'RUA2', 'RUA3', "REJC", "RFAradius", "RFAulna"],
    #
    #     "T8": ["T8", "RASH", "LASH", "CLAV", "LACR", "LSJC",'LUA1', 'LUA2', 'LUA3', "LEJC", "LFAradius", "LFAulna", "RACR", "RSJC",'RUA1', 'RUA2', 'RUA3', "REJC", "RFAradius", "RFAulna"]
    #
    # }

    xsens_markers = {

        "LeftForeArm": ["LEJC", "LFAradius", "LFAulna", "LFAsuperior"],
        "RightForeArm": ["REJC", "RFAradius", "RFAulna", "RFAsuperior"],
        "LeftUpperArm": ["LSJC", 'LUA1', 'LUA2', 'LUA3', "LEJC", "LLEL", "LMEL"],
        "RightUpperArm": ["RSJC", 'RUA1', 'RUA2', 'RUA3', "REJC", "RLEL", "RMEL"],
        "LeftUpperLeg": ["LHJC", 'LTH1', 'LTH2', 'LTH3', "LLFC", "LMFC", "LKJC", "L_tibial_plateau"],
        "RightUpperLeg": ["RHJC", 'RTH1', 'RTH2', 'RTH3', "RLFC", "RMFC", "RKJC", "R_tibial_plateau"],
        #"LeftUpperArm": ["LSJC", 'LUA1', 'LUA2', 'LUA3', "LEJC", "LFAradius", "LFAulna", "LLEL", "LMEL", "LFAsuperior"],
        # "RightUpperArm": ["RSJC", 'RUA1', 'RUA2', 'RUA3', "REJC", "RLEL", "RMEL", "RFAradius", "RFAulna"],

        # "LeftShoulder": ["LACR", "LSJC",'LUA1', 'LUA2', 'LUA3', "LEJC", "LFAradius", "LFAulna"],
        # "RightShoulder": ["RACR", "RSJC",'RUA1', 'RUA2', 'RUA3', "REJC", "RFAradius", "RFAulna"],

        # "T8": ["T8", "RASH", "RPSH", "LASH","LPSH", "CLAV", "LACR", "LSJC", 'LUA1', 'LUA2', 'LUA3', "LEJC", "LFAradius",
        #        "LFAulna", "RLEL", "RMEL", "RFAsuperior", "LLEL", "LMEL", "LFAsuperior",
        #        "RACR", "RSJC", 'RUA1', 'RUA2', 'RUA3', "REJC", "RFAradius", "RFAulna"]

        "T8": ["T8", "RASH", "RPSH", "LASH","LPSH", "CLAV", "LACR", "LSJC", "RACR", "RSJC"],
        "Pelvis": ["T8", "RASI", "LASI", "RPSI", "LPSI", "RHJC", "LHJC", "Sacrum"]

    }


def left_upper_leg(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df

def right_upper_leg(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df

def left_upper_arm(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df

    # ml = Param.xsens_markers['LeftForeArm'][1:]
    # add = np.array([bob['LEJC'] for j in range(0, bob[ml].shape[1])]).T
    # ref_m = osmarkers['LEJC'].to_numpy()
    # sub = np.array([-1 * ref_m for j in range(0, bob[ml].shape[1])]).T
    # bob[ml] = bob[ml] + sub + add

def left_forearm(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df

def right_upper_arm(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df

    # ml = Param.xsens_markers['RightForeArm'][1:]
    # add = np.array([bob['REJC'] for j in range(0, bob[ml].shape[1])]).T
    # ref_m = osmarkers['REJC'].to_numpy()
    # sub = np.array([-1 * ref_m for j in range(0, bob[ml].shape[1])]).T
    # bob[ml] = bob[ml] + sub + add

def right_forearm(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df


def right_upper_leg(bob, marker):
    mox = copy.deepcopy(bob[marker_list])
    ref_m = bob[marker].to_numpy()
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T

    lk = mox - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, -180, 0], degrees=True)

    moy2b = r.apply(pik)
    moy2b = (orient_data[ot][i]).apply(moy2b)
    r0 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True)
    moy2b = r0.apply(moy2b)
    moy2b = r.inv().apply(moy2b)
    moy2 = moy2b.T + default_pos  # + ref

    moy_df = pd.DataFrame(data=moy2, columns=mo.columns)
    bob[marker_list] = moy_df


def T8(bob):
    marker_list = Param.xsens_markers[ot]
    mo = copy.deepcopy(bob[marker_list])
    imu_idx = Param.xsens_joints['IMU'].index(ot)
    marker = Param.xsens_joints['Marker'][imu_idx]
    # ref_m = mo[marker].to_numpy()
    ref_m = osmarkers[marker].to_numpy()
    marker_position_label = [m for m in df_markers.columns if marker in m]

    # mox = copy.deepcopy(osmarkers[marker_list])
    current = df_markers[marker_position_label].to_numpy()[i, :]
    # cur = np.array([current for j in range(0, mox.shape[1])]).T
    cur = np.atleast_2d(current)
    #default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T
    default_pos = np.atleast_2d(ref_m).T
    lk = mo - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True)

    moy2a = r.apply(pik)
    moy2b = (orient_data2[ot][i]).apply(moy2a)
    moy2b = r.inv().apply(moy2b).T + default_pos + cur

    moy_df = pd.DataFrame(data=moy2b, columns=mo.columns)
    bob[marker_list] = moy_df

    effects = ['LeftUpperArm', 'RightUpperArm', 'LeftForeArm', 'RightForeArm']

    for e in effects:
        pp = Param.xsens_joints['IMU']
        pq = Param.xsens_joints['Marker']
        point = pq[pp.index(e)]
        ml = Param.xsens_markers[e][1:]
        add = np.array([bob[point] for j in range(0, bob[ml].shape[1])]).T
        ref_m = osmarkers[point].to_numpy()
        sub = np.array([-1*ref_m for j in range(0, bob[ml].shape[1])]).T
        bob[ml] = bob[ml] + sub +add

    pass

    # colm = [cm for cm in mo.columns]
    # moy_df = pd.DataFrame(data=moy2b, columns=colm)


    return bob


def Pelvis(bob, marker):
    marker_list = Param.xsens_markers[ot]
    mo = copy.deepcopy(bob[marker_list])
    ox = osmarkers['T8']

    mo['T8'] = ox
    imu_idx = Param.xsens_joints['IMU'].index(ot)
    #marker = Param.xsens_joints['Marker'][imu_idx]
    # ref_m = mo[marker].to_numpy()
    ref_m = osmarkers[marker].to_numpy()
    marker_position_label = [m for m in df_markers.columns if marker in m]

    mox = copy.deepcopy(osmarkers[marker_list])
    current = df_markers[marker_position_label].to_numpy()[i, :]
    cur = np.array([current for j in range(0, mox.shape[1])]).T
    default_pos = np.array([ref_m for j in range(0, mox.shape[1])]).T
    lk = mo - default_pos

    pik = [lk.iloc[:, o] for o in range(0, lk.shape[1])]
    r = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True)

    moy2a = r.apply(pik)
    moy2b = (orient_data2[ot][i]).apply(moy2a)
    moy2b = r.inv().apply(moy2b).T # + default_pos + cur

    moy_df = pd.DataFrame(data=moy2b, columns=mo.columns)
    effects = [['T8', ['LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm']],
               'LeftUpperLeg', 'RightUpperLeg']

    bob[marker_list] = moy_df


    for e in effects:
        ex = e
        if isinstance(e, list):
            ex = e[0]
        pp = Param.xsens_joints['IMU']
        pq = Param.xsens_joints['Marker']
        point = pq[pp.index(ex)]
        ml = Param.xsens_markers[ex][1:]
        bob_root = copy.deepcopy(bob[ex])
        # add = np.array([moy_df[point] for j in range(0, bob[ml].shape[1])]).T
        ref_m = copy.deepcopy(bob[point].to_numpy())
        # sub = np.array([-1 * ref_m for j in range(0, bob[ml].shape[1])]).T
        # bob[ml] = bob[ml] + sub + add
        bob[ml] = bob[ml] - np.atleast_2d(ref_m).T + np.atleast_2d(moy_df[point]).T
        if isinstance(e, list):
            for f in e[1]:
                mlx = Param.xsens_markers[f][1:]
                bob[mlx] = bob[mlx] - np.atleast_2d(ref_m).T + np.atleast_2d(moy_df[point]).T
        bob[ex] = bob_root


    return bob





if __name__ == "__main__":
    test = "M:/Mocap/Movella_Re/P025/Straight Normal 01/LeftForeArm_imu_vec3_2.csv"
    est = "M:/Mocap/Movella_Re/P025/Straight Normal 01/"
    e = [f for f in os.listdir(est) if f.split('_')[0] in Param.xsens_joints['IMU'] and f.endswith('vec3_2.csv') or f.endswith('vec3.csv')]
    eo = [f for f in os.listdir(est) if
          f.split('_')[0] in Param.xsens_joints['IMU'] and f.endswith('imu_ori.csv')]
    out_folder = "I:/Meta/IMU/P025/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    tilename = "{0}/Straight Normal 01_imu.trc".format(out_folder)
    osim = OsimHelper(r"I:\Meta\IMU\P025\P025_RajagopaXsens_scaled.osim")
    neutral = pd.read_csv(r"I:\Meta\IMU\P025\neutal_pose.csv")
    v = neutral.iloc[:, 1:]
    # setting initial pose
    osmarkers_before = copy.deepcopy(osim.markerset)
    osim.set_joints(v)
    osmarkers = copy.deepcopy(osim.markerset)
    b_p = osmarkers_before[Param.xsens_markers['Pelvis']]
    p_p = osmarkers[Param.xsens_markers['Pelvis']]
    r = Cloud.rigid_body_transform(p_p.to_numpy(), b_p.to_numpy())
    rr = Rotation.from_matrix(r[:3, :3])
    rrr = rr.as_euler('xyz', degrees=True)
    # osmarkersx = osmarkers.to_numpy() + np.atleast_2d(r[:3, 3]).T
    osmarkersx = np.matmul(r[:3, :3], osmarkers.to_numpy()) + np.atleast_2d(r[:3, 3]).T
    osmarkers = pd.DataFrame(data=osmarkersx, columns=osmarkers.columns)
    marker_labels = [c for c in osmarkers.columns]
    markers_df_col = ['#Frame', 'Time']
    midx = 1
    marker_map = {}
    for m in marker_labels:
        markers_df_col.append('{0}_X{1}'.format(m, midx))
        markers_df_col.append('{0}_Y{1}'.format(m, midx))
        markers_df_col.append('{0}_Z{1}'.format(m, midx))
        marker_map[m] = ['{0}_X{1}'.format(m, midx), '{0}_Y{1}'.format(m, midx), '{0}_Z{1}'.format(m, midx)]
        midx +=1
    pos = ['m_position_X', 'm_position_Y', 'm_position_Z']
    timex = ['time']
    idx = 1
    cols = ['#Frame', 'Time']
    time_np = None
    marker_data = []

    for ef in e:
        filen = "{0}{1}".format(est, ef)
        imu = os.path.split(filen)[1].split("_")[0]
        imu_idx = Param.xsens_joints['IMU'].index(imu)
        marker = Param.xsens_joints['Marker'][imu_idx]
        df = pd.read_csv(filen)
        pos_df = df[pos]
        time_series = df[timex]
        time_series = time_series - time_series.iloc[0, 0]
        if time_np is None:
            time_np = time_series.to_numpy()[:, 0]

        marker_data.append(pos_df.to_numpy())

        cols.append('{0}_X{1}'.format(marker, idx))
        cols.append('{0}_Y{1}'.format(marker, idx))
        cols.append('{0}_Z{1}'.format(marker, idx))
        idx += 1

    np_markers = np.hstack(marker_data)
    npx = np.zeros([time_np.shape[0], np_markers.shape[1] + 2])
    npx[:, 0] = [i + 1 for i in range(0, npx.shape[0])]
    npx[:, 1] = time_np
    npx[:, 2:] = np_markers
    df_markers = pd.DataFrame(data=npx, columns=cols)

    osm_df = pd.DataFrame(data=np.zeros([time_np.shape[0], len(markers_df_col)]), columns=markers_df_col)

    orient_data = {}
    orient_data2 = {}
    for ef in eo:
        filen = "{0}{1}".format(est, ef)
        imu = os.path.split(filen)[1].split("_")[0]
        print(imu)
        imu_idx = Param.xsens_joints['IMU'].index(imu)
        marker = Param.xsens_joints['Marker'][imu_idx]
        df = pd.read_csv(filen)
        df.iloc[:, 0] = df.iloc[:, 0] - df.iloc[0, 0]
        ref = Rotation.from_quat(df.iloc[0, 1:], scalar_first=True).inv()
        # ref = Rotation.from_euler('xyz', [-90, 0, 0])
        # orient = [np.matmul(Rotation.from_quat(df.iloc[i, 1:], scalar_first=True).as_matrix(), ref) for i in
        #           range(0, df.shape[0])]

        orient = [Rotation.from_quat(df.iloc[i, 1:], scalar_first=True) * ref for i in
                  range(0, df.shape[0])]
        orient2 = [Rotation.from_quat(df.iloc[i, 1:], scalar_first=True)for i in range(0, df.shape[0])]
        orient_data[imu] = orient
        orient_data2[imu] = orient2
        pass

    for i in range(0, df_markers.shape[0]):
        bob = copy.deepcopy(osmarkers)
        print(i)
        imu_order = ['RightUpperLeg',  'RightForeArm', 'RightUpperArm', 'LeftForeArm', 'LeftUpperArm', 'T8', 'Pelvis']
        for ot in imu_order:
            try:
                marker_list = Param.xsens_markers[ot]
                mo = copy.deepcopy(bob[marker_list])
                imu_idx = Param.xsens_joints['IMU'].index(ot)
                marker = Param.xsens_joints['Marker'][imu_idx]

                if 'LeftForeArm' in ot:
                    left_forearm(bob, marker)
                if 'RightForeArm' in ot:
                    right_forearm(bob, marker)
                if 'LeftUpperArm' in ot:
                    left_upper_arm(bob, marker)
                if 'RightUpperArm' in ot:
                    right_upper_arm(bob, marker)
                if 'LeftUpperLeg' in ot:
                    left_upper_leg(bob, marker)
                if 'RightUpperLeg' in ot:
                    right_upper_leg(bob, marker)
                if 'T8' in ot:
                    T8(bob)
                if 'Pelvis' in ot:
                    bob = Pelvis(bob, marker)


            except KeyError:
                continue
            pass
        for c in bob.columns:
            cols = [markers_df_col.index(l) for l in marker_map[c]]
            d = bob[c].to_numpy()
            osm_df.iloc[i, cols] = d
            pass
        osm_df.iloc[i, 0] = i+1
        osm_df.iloc[i, 1] = time_np[i]
        pass


    # trc = TRC.create_from_panda_dataframe(df_markers, tilename)
    # trc.z_up_to_y_up()
    # trc.write(tilename)
    print('write out')
    trc = TRC.create_from_panda_dataframe(osm_df, tilename)
    trc.write(tilename)
    pass