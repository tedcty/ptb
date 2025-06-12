from ptb.util.data import MocapDO
from ptb.util.osim.osim_store import OSIMForcePlate, OSIMStorage, HeadersLabels
import os
import numpy as np
import pandas as pd
import time
import ptb


if __name__ == "__main__":
    # d = './walk20.mot'
    # w = './walk20a.mot'
    # osim_mot = OSIMStorage.read(d)
    # osim_mot.write(w)
    print("read")
    print(ptb.__version__)
    start = time.time()
    f = 'I:/walking_speed_NoAFO_Ella.c3d'
    m = MocapDO.create_from_c3d(f)
    print(time.time() - start)
    m.z_up_to_y_up()
    print(time.time() - start)
    m.export_opensim_mot_force_file('I:/')
    print(time.time() - start)
    # header = OSIMStorage.simple_header_template()
    # header[HeadersLabels.trial] = os.path.split(f)[1][:os.path.split(f)[1].rindex('.')]
    # header[HeadersLabels.nRows] = m.force_plates.data.shape[0]
    # header[HeadersLabels.nColumns] = m.force_plates.num_of_plates*9+1
    # header[HeadersLabels.inDegrees] = True
    # data = np.zeros([header[HeadersLabels.nRows], header[HeadersLabels.nColumns]])
    # osimcols = ['time']
    # data[:, 0] = m.force_plates.x
    # plates = []
    # unit = 1
    # if m.force_plates.units["COP"]["COP.Px1"] == 'mm':
    #     unit = 1000
    #
    # for p in m.force_plates.plate:
    #     xf = m.force_plates.plate[p][1]
    #     idx = m.force_plates.plate[p][0]
    #     plate = [OSIMForcePlate.ground_force, OSIMForcePlate.ground_torque]
    #     info_plate = []
    #     for d in plate:
    #         k = d.generate_label(m.force_plates.plate[p][0])
    #         l = OSIMForcePlate.map_from_c3d(d, idx)
    #         if d == OSIMForcePlate.ground_force:
    #             v_df = pd.DataFrame(data=xf[l["v"]].to_numpy(), columns=k["v"])
    #             p_df = pd.DataFrame(data=xf[l["p"]].to_numpy()/unit, columns=k["p"])
    #             f = pd.concat([v_df, p_df], axis=1)
    #             info_plate.append(f)
    #         else:
    #             info_plate.append(pd.DataFrame(data=xf[l].to_numpy(), columns=k['d']))
    #         pass
    #     g = pd.concat(info_plate, axis=1)
    #     plates.append(g)
    # h = pd.concat(plates, axis=1)
    # data[:, 1:] = h.to_numpy()
    # osimcols = [c for c in h.columns]
    # osimcols.insert(0, 'time')
    # data_df = pd.DataFrame(data=data, columns=osimcols)
    # w = "I:/{0}_test.mot".format(header[HeadersLabels.trial])
    # osim_mot = OSIMStorage.create(data_df, header, w)
    # osim_mot.write(w)
    # print(time.time() - start)
    # parts = ['v', 'p']
    # i = 1
    # osim_mot_data = []
    # for p in plates:
    #     pm = p[0]
    #     plate_group = []
    #     for d in parts:
    #         labels = pm[OSIMForcePlate.ground_force][1]
    #         col = ['{0}{1}'.format(c, i) for c in labels[d]]
    #         x = p[1][col]
    #         for c in pm[OSIMForcePlate.ground_force][0][d]:
    #             osimcols.append(c)
    #         df = pd.DataFrame(data=x.to_numpy(), columns=pm[OSIMForcePlate.ground_force][0][d])
    #         plate_group.append(df)
    #     labels = pm[OSIMForcePlate.ground_torque][1]
    #     col = ['{0}{1}'.format(c, i) for c in labels]
    #     x = p[1][col]
    #     for c in pm[OSIMForcePlate.ground_torque][0]['d']:
    #         osimcols.append(c)
    #     df = pd.DataFrame(data=x.to_numpy(), columns=pm[OSIMForcePlate.ground_torque][0]['d'])
    #     plate_group.append(df)
    #     osim_mot_data.append(plate_group)
    #     i += 1
    #
    # data_df = pd.DataFrame(data=data, columns=osimcols)
    # for d in osim_mot_data:
    #     for e in d:
    #         col = [c for c in e.columns]
    #         data_df[col] = e.to_numpy()
    #         pass
    #
    # w = "I:/{0}_test.mot".format(header[HeadersLabels.trial])
    # osim_mot = OSIMStorage.create(data_df, header, w)
    # osim_mot.write(w)
    # print(time.time()-start)
    pass