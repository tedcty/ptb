from ptb.util.data import MocapDO
from ptb.util.osim.osim_store import OSIMForcePlate, OSIMStorage, HeadersLabels
import os
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # d = './walk20.mot'
    # w = './walk20a.mot'
    # osim_mot = OSIMStorage.read(d)
    # osim_mot.write(w)
    print("read")
    f = 'D:/walking_speed_NoAFO_Ella.c3d'
    m = MocapDO.create_from_c3d(f)
    m.z_up_to_y_up()
    header = OSIMStorage.simple_header_template()
    header[HeadersLabels.trial] = os.path.split(f)[1][:os.path.split(f)[1].rindex('.')]
    header[HeadersLabels.nRows] = m.force_plates.data.shape[0]
    header[HeadersLabels.nColumns] = m.force_plates.num_of_plates*9+1
    header[HeadersLabels.inDegrees] = True
    data = np.zeros([header[HeadersLabels.nRows], header[HeadersLabels.nColumns]])
    osimcols = ['time']
    data[:, 0] = m.force_plates.x
    plates = []
    for p in m.force_plates.plate:
        xf = m.force_plates.plate[p][1]
        plate = {OSIMForcePlate.ground_force: None, OSIMForcePlate.ground_torque: None}
        for d in plate:
            k = d.generate_label(m.force_plates.plate[p][0])
            plate[d] = [k, OSIMForcePlate.map_from_c3d(d)]
            pass
        plates.append([plate, xf])
    parts = ['v', 'p']
    i = 1
    osim_mot_data = []
    for p in plates:
        pm = p[0]
        plate_group = []
        for d in parts:
            labels = pm[OSIMForcePlate.ground_force][1]
            col = ['{0}{1}'.format(c, i) for c in labels[d]]
            x = p[1][col]
            for c in pm[OSIMForcePlate.ground_force][0][d]:
                osimcols.append(c)
            df = pd.DataFrame(data=x.to_numpy(), columns=pm[OSIMForcePlate.ground_force][0][d])
            plate_group.append(df)
        labels = pm[OSIMForcePlate.ground_torque][1]
        col = ['{0}{1}'.format(c, i) for c in labels]
        x = p[1][col]
        for c in pm[OSIMForcePlate.ground_torque][0]['d']:
            osimcols.append(c)
        df = pd.DataFrame(data=x.to_numpy(), columns=pm[OSIMForcePlate.ground_torque][0]['d'])
        plate_group.append(df)
        osim_mot_data.append(plate_group)
        i += 1

    data_df = pd.DataFrame(data=data, columns=osimcols)
    for d in osim_mot_data:
        pass
    pass