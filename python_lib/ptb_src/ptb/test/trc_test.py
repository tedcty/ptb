from ptb.util.io.mocap.file_formats import TRC
import pandas as pd
import numpy as np
import os


xsens = {
    "Marker": ["LPSH", "RPSH", "LUA3", "RUA3", "LFAsuperior", "RFAsuperior", "Sacrum", "CLAV", "LTH1", "RTH1", "LTB3", "RTB3", "LMT5", "RMT5"],
    "IMU": ["LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm", "LeftForeArm", "RightForeArm", "Pelvis", "T8", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot"]
}


if __name__ == "__main__":
    test = "M:/Mocap/Movella_Re/P025/Straight Normal 01/LeftForeArm_imu_vec3_2.csv"
    tilename = "M:/Mocap/Movella_Re/P025/Straight Normal 01_imu.trc"
    imu = os.path.split(test)[1].split("_")[0]
    imu_idx = xsens['IMU'].index(imu)
    marker = xsens['Marker'][imu_idx]
    df = pd.read_csv(test)
    pos = ['m_position_X', 'm_position_Y', 'm_position_Z']
    timex = ['time']
    idx = 1
    pos_df = df[pos]
    time_series = df[timex]
    time_series = time_series - time_series.iloc[0, 0]
    npdf = np.zeros([time_series.shape[0], 5])
    npdf[:, 1] = time_series.to_numpy()[:, 0]
    npdf[:, 2:] = pos_df.to_numpy()
    npdf[:, 0] = [i+1 for i in range(0, time_series.shape[0])]
    cols = ['Frame', 'Time', '{0}_X{1}'.format(marker, idx), '{0}_Y{1}'.format(marker, idx), '{0}_Z{1}'.format(marker, idx)]
    df_markers = pd.DataFrame(data=npdf, columns=cols)
    trc = TRC.create_from_panda_dataframe(df_markers, tilename)
    pass