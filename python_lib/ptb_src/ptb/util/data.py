from ptb.util.io.helper import *
from ptb.util.io.mesh import *
from ptb.util.io.mocap.file_formats import *
from ptb.util.math.filters import *
from ptb.util.math.stat import *

from typing import Optional

from datetime import datetime, timezone
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from ptb.util.osim.osim_store import OSIMStorage, OSIMForcePlate, HeadersLabels

"""
TODOs:
> Add Documentation to code
"""



def stamp():
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
    return dt_string, now


def date_time_convert(date_str):
    return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")

def date_utcnow():
    return datetime.now(timezone.utc)

def milli():
    date = date_utcnow().replace(tzinfo=None) - datetime(1970, 1, 1)
    seconds = (date.total_seconds())
    milliseconds = round(seconds * 1000)
    return milliseconds

class MocapDO:
    def __init__(self):
        self.__raw__ = None
        self.markers: TRC = None
        self.force_plates = None
        self.emg = None
        self.imu = None
        self.other_analog = None
        self.filepath = ""

    @property
    def marker_set(self):
        return self.markers.marker_set

    def z_up_to_y_up(self):
        r = self.markers.z_up_to_y_up()
        self.force_plates.rotate(r)

    def export_opensim_mot_force_file(self, out_dir):
        f = self.filepath
        header = OSIMStorage.simple_header_template()
        header[HeadersLabels.trial] = os.path.split(f)[1][:os.path.split(f)[1].rindex('.')]
        header[HeadersLabels.nRows] = self.force_plates.data.shape[0]
        header[HeadersLabels.nColumns] = self.force_plates.num_of_plates * 9 + 1
        header[HeadersLabels.inDegrees] = True
        data = np.zeros([header[HeadersLabels.nRows], header[HeadersLabels.nColumns]])
        data[:, 0] = self.force_plates.x
        plates = []
        unit = 1
        if self.force_plates.units["COP"]["COP.Px1"] == 'mm':
            unit = 1000

        for p in self.force_plates.plate:
            xf = self.force_plates.plate[p][1]
            idx = self.force_plates.plate[p][0]
            plate = [OSIMForcePlate.ground_force, OSIMForcePlate.ground_torque]
            info_plate = []
            for d in plate:
                k = d.generate_label(self.force_plates.plate[p][0])
                l = OSIMForcePlate.map_from_c3d(d, idx)
                if d == OSIMForcePlate.ground_force:
                    v_df = pd.DataFrame(data=xf[l["v"]].to_numpy(), columns=k["v"])
                    p_df = pd.DataFrame(data=xf[l["p"]].to_numpy() / unit, columns=k["p"])
                    f = pd.concat([v_df, p_df], axis=1)
                    info_plate.append(f)
                else:
                    info_plate.append(pd.DataFrame(data=xf[l].to_numpy(), columns=k['d']))
                pass
            g = pd.concat(info_plate, axis=1)
            plates.append(g)
        h = pd.concat(plates, axis=1)
        data[:, 1:] = h.to_numpy()
        osimcols = [c for c in h.columns]
        osimcols.insert(0, 'time')
        data_df = pd.DataFrame(data=data, columns=osimcols)
        w = "{0}{1}_ptb.mot".format(out_dir, header[HeadersLabels.trial])
        osim_mot = OSIMStorage.create(data_df, header, w)
        osim_mot.write(w)
        return osim_mot

    @staticmethod
    def create_from_c3d(file):
        # Optimized to read file only once
        sg = StorageIO.readc3d_general(file)
        
        m = MocapDO()
        m.filepath = file
        
        # Populate markers directly from the dictionary returned by readc3d_general
        # We need to adapt the caching structure expected by TRC.create_from_c3d_dict
        # The 'sg' dict from readc3d_general is already compatible with create_from_c3d_dict
        # but we need to ensure the structure matches exactly what TRC expects.
        # TRC.create_from_c3d_dict expects a dict with 'mocap', 'markers', etc.
        
        # NOTE: StorageIO.readc3d_general returns a dict but it is slightly different 
        # from what simple_readc3d returned. Simple_readc3d returned:
        # {'markers': ..., 'mocap': ..., 'text': ..., 'analog': ...}
        # read_c3d_general returns:
        # {'analog_channels_label': ..., 'point_data': ..., 'analog_data': ..., 'markers': ...}
        
        # We need to construct the 's' dict from 'sg' to avoid re-reading.
        
        # 1. Reconstruct 'mocap' header info for TRC
        mocap_header = sg['mocap'] # readc3d_general already has this key
        
        # 2. Reconstruct 'markers' dict for TRC
        # readc3d_general does NOT strictly have 'markers' in the standard sense of simple_readc3d
        # It has 'point_data' DataFrame. We need to convert this back or update TRC to use point_data?
        # Actually, looking at readc3d_general source, it DOES NOT create a 'markers' dict key by default.
        # But wait, looking at `Yac3do.trc` property in this file (lines 142+), it constructs 'markers'
        # from 'point_data'. We should do the same here to reuseTRC.create_from_c3d_dict
        
        markers = {}
        # markers['time'] already usually handled, but let's be safe
        # point_data has columns frame, time, marker_X, marker_Y, marker_Z...
        
        # Efficiently extract markers from point_data DataFrame
        df_points = sg['point_data']
        # Get unique marker names (stripping _X, _Y, _Z)
        # sg['point_label'] has the raw names
        
        # Construct the markers dictionary expected by TRC.create_from_c3d_dict
        # It expects {marker_name: [list of [x,y,z] or array], ...} 
        # TRC.create_from_c3d_dict (line 187 file_formats.py) does:
        # markers = c3d_dic["markers"]
        # markers_np = {n: np.array(markers[n]) for n in markers}
        # So it expects a list or array of shape (N, 3).
        
        cols = df_points.columns
        for lbl in sg['point_label']:
            # Assuming labels in point_data are "Label_X", "Label_Y", "Label_Z"
            if f"{lbl}_X" in cols:
                # Extract as Nx3 array
                dat = df_points[[f"{lbl}_X", f"{lbl}_Y", f"{lbl}_Z"]].to_numpy()
                markers[lbl] = dat
        
        # Add time if needed, TRC checks it (line 195 file_formats.py) uses markers_np[marker_labels[0]] ??
        # Actually TRC line 195: times = markers_np[marker_labels[0]] ... wait
        # If marker_labels[0] is 'time', then it expects 'time' key.
        # In simple_readc3d, 'time' key is added.
        markers['time'] = df_points['time'].to_numpy()
        
        # Construct the 's' dictionary mimicking simple_readc3d output
        s_reconstructed = {
            'mocap': mocap_header,
            'markers': markers,
            # 'analog': ... TRC doesn't seem to use analog, MocapDO handles it below
        }
        
        m.markers = TRC.create_from_c3d_dict(s_reconstructed, file)
        
        # Now handle Force Plates
        paramF = {'corners': sg['force_plate_corners'],
                  'origin': sg['force_plate_origins_from_corner'],
                  'num_plates': sg['number_of_force_plates']
                  }
        
        ad = sg["analog_data"]
        if ad is None or ad.empty:
            m.force_plates = None
            return m

        # Robust Column Identification
        # Instead of generic 'Force', 'Moment', look for explicit Fx, Fy, Fz etc patterns
        # or use the channels defined in the C3D force platform section if available.
        # sg['force_plate_channels'] might contain indices?
        
        # Let's try a robust regex-like filter first
        all_cols = ad.columns
        
        # Helper to find columns related to Force/Moment
        def matches_force_moment(col_name):
            c = col_name.lower()
            return 'force' in c or 'moment' in c or 'fx' in c or 'fy' in c or 'fz' in c or 'mx' in c or 'my' in c or 'mz' in c

        ad_col = [c for c in all_cols if matches_force_moment(c) or 'frame' in c.lower() or 'time' in c.lower()]
        
        # Attempt to map units safely
        forces_unit = {}
        moment_unit = {}
        
        # sg["analog_unit"] is a dict {channel_name: unit_string}
        # We need to map these to the 'Force', 'Moment' categories.
        # The previous code assumed strict naming "Force.Fx1" etc.
        # Now we might have simpler names.
        
        for c in ad_col:
            if c in sg["analog_unit"]: # Strict match
                u = sg["analog_unit"][c]
                if 'force' in c.lower() or 'f' in c.lower(): 
                   forces_unit[c] = u
                if 'moment' in c.lower() or 'm' in c.lower():
                    moment_unit[c] = u
        
        cop_unit = {}

        # Safe unit extraction
        for i in moment_unit:
            unit = moment_unit[i]
            # Robust split
            if 'N' in unit and 'm' in unit:
                 # Likely Nmm or Nm
                 # naive replacement logic from before:
                 # dk = unit.split('N') -> assumes "N" is the separator
                 # If unit is "Nmm", split('N') -> ['', 'mm'] -> "Nmm"
                 # If unit is "Nm", split('N') -> ['', 'm'] -> "Nm"
                 # It was trying to construct pressure unit? "N" -> "P"?
                 # Pre-existing logic: element 0 becomes "N", element 1 is suffix. 
                 # i.e. "Nmm" -> dk=['', 'mm'] -> dk[0]="N" -> result "mm"
                 # It seems it wants the distance unit part of the moment.
                 dist_unit = unit.replace('N', '').replace('n', '').strip()
                 cop_unit_key = i.replace('Moment', 'COP').replace('M', 'P') # Naive string replace
                 cop_unit[cop_unit_key] = dist_unit
            else:
                 # Fallback
                 cop_unit[i] = "mm"

        # Ensure time is present
        if 'time' not in ad_col:
             ad_col.insert(0, 'time')

        # Subset analog data
        # Check if columns exist (clean up ad_col)
        ad_col = [c for c in ad_col if c in ad.columns]
        
        ad_force = ad[ad_col]
        
        # If time is missing in data but we need it (should have been from readc3d_general logic)
        # data.py previous logic created it manually:
        # p[:, 0] = [i*(1/sg['analog_rate']) for i in range(ad_force.shape[0])]
        # read_c3d_general creates 'time' column in sub-frame loop? No, it creates 'sub-frame'.
        # Actually it creates 'mocap-frame', 'sub-frame' and then analog data.
        # We need to calculate time.
        
        # Create a new dataframe with time explicitly
        force_data = ad_force.copy()
        
        # Calculate time: (frame_idx * subframes + subframe_idx) * dt?
        # Or simpler: just index * (1/analog_rate)
        time_array = np.arange(force_data.shape[0]) * (1.0 / sg['analog_rate'])
        # Add start time offset if needed? sg['first_frame']?
        # Usually 0-based relative time is sufficient for many pipelines, but let's be consistent
        # previous code: p[:, 0] = [i*(1/sg['analog_rate']) ...
        
        force_data.insert(0, 'time', time_array)
        
        # Create ForcePlate object
        f = ForcePlate.create(paramF, force_data)
        f.units = {'Force': forces_unit, 'Moment': moment_unit, 'COP': cop_unit}
        m.force_plates = f
        
        return m


class Yac3do:
    # Yet another c3d data object
    #
    def __init__(self, filename: str=None):
        self.filename = filename
        self.c3d_dict = StorageIO.readc3d_general(filename)
        self.__trc__ = None
        self.__trc__dirty__ = True
        pass

    @property
    def trc(self):
        if self.__trc__dirty__ or self.__trc__ is None:
            ret = self.c3d_dict['point_data'].copy()
            lab = ['time']
            for m in self.marker_labels:
                lab.append(m)
            markers = {m: [] for m in lab}
            for m in lab:
                if m == 'time':
                    markers[m] = [n*(1/self.c3d_dict['meta_data']['trial']['camera_rate']) for n in range(0, self.c3d_dict['meta_data']['trial']['end_frame'])]
                else:
                    columns = [n for n in ret.columns if m in n]
                    markers[m] = ret[columns]
                pass
            self.c3d_dict['markers'] = markers
            self.__trc__ = TRC.create_from_c3d_dict(self.c3d_dict, self.filename)
            self.__trc__dirty__ = False
        return self.__trc__

    @property
    def marker_labels(self):
        return self.c3d_dict['point_label']

    @property
    def markers(self):
        frames = [i+1 for i in range(0, self.c3d_dict['point_data'].shape[0])]
        t = [i*(1/self.c3d_dict['point_rate']) for i in range(0, self.c3d_dict['point_data'].shape[0])]
        ret = self.c3d_dict['point_data'].copy()
        ret.insert(0, 'frames', frames)
        ret.insert(1, 'time', t)
        return ret

    @property
    def analog(self):
        return self.c3d_dict['analog_data']


class Yadict(object):
    # Yet another dictionary
    # provides core set and get dict
    # and get keys in a list
    def __init__(self, d: dict):
        self.d = d

    def __getitem__(self, item):
        return self.d[item]

    def __setitem__(self, item, value):
        self.d[item] = value

    def get_keys(self):
        return [k for k in self.d]

class Mesh:
    def __init__(self, filename=None, ignore=False):
        self.mesh_filename = filename
        self.actor = None
        self.mesh_tool = VTKMeshUtl.get_type(filename)
        self.reader = self.mesh_tool.reader()
        self.current_reader = self.mesh_tool.label()

        self.mesh = None
        self.__points__ = None
        self.cog = None
        self.mc = None
        self.volume = None
        self.ignore_check = ignore
        if self.mesh_filename is not None:
            self.load_as_vtk(self.mesh_filename)

    @property
    def points(self):
        return self.__points__

    @points.setter
    def points(self, p):
        if isinstance(p, vtk.vtkPolyData):
            self.__points__ = VTKMeshUtl.extract_points(p)
        elif isinstance(p, np.ndarray):
            self.__points__ = p
        else:
            self.__points__ = None

    @staticmethod
    def convert_vtp_2_stl(vtp_file, stl_out):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_file)
        reader.Update()

        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetFileName(stl_out)
        stl_writer.SetInputConnection(reader.GetOutputPort())
        stl_writer.Write()

    @staticmethod
    def convert_vtp_2_vtp(vtp_file, stl_out):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_file)
        reader.Update()

        vtp_writer = vtkXMLPolyDataWriter()
        vtp_writer.SetFileName(stl_out)
        vtp_writer.SetDataModeToAscii()
        vtp_writer.SetInputConnection(reader.GetOutputPort())
        vtp_writer.Write()

    @staticmethod
    def convert(in_file, out_file):
        in_poly = VTKMeshUtl.load(in_file)
        VTKMeshUtl.write(out_file, in_poly)

    @staticmethod
    def convert_batch(working_dir: str, to_ext: str = ".ply"):
        if os.path.exists(working_dir):
            if not working_dir.endswith("/") or working_dir.endswith("\\"):
                working_dir += "/"
            wr_list = [f for f in os.listdir(working_dir) if VTKMeshUtl.is_valid(working_dir+f)]
            for w in wr_list:
                en = w.rindex(".")
                Mesh.convert(working_dir + w, working_dir + w[:en] + to_ext)

    @staticmethod
    def convert_vtp_2_stl_batch(wr):
        wr_list = os.listdir(wr)
        for w in wr_list:
            en = w.rindex(".")
            print(w[:en])
            Mesh.convert_vtp_2_stl(wr + w[:en] + '.vtp', wr + w[:en] + '.stl')

    def load_as_vtk(self, filename):
        self.mesh_filename: str = filename
        self.mesh_tool = VTKMeshUtl.get_type(self.mesh_filename)
        self.reader = self.mesh_tool.reader()
        self.mesh = VTKMeshUtl.load(self.mesh_filename)
        self.cog = VTKMeshUtl.cog(self.mesh)
        self.volume = VTKMeshUtl.volume(self.mesh)
        self.mc = self.cog
        self.points = VTKMeshUtl.extract_points(self.mesh)
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            # mapper.SetInput(reader.GetOutput())
            mapper.SetInput(self.mesh)
        else:
            mapper.SetInputData(self.mesh)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)

    @property
    def center_of_gravity(self):
        return self.cog

    @property
    def mesh_center(self):
        return self.mc

    def apply_transform(self, m):
        axis = 0
        if self.points.shape[0] == 3:
            axis = 1
        dummy = np.ones([4, self.points.shape[axis]])

        if self.points.shape[0] == 3:
            dummy[:3, :] = self.points
        else:
            dummy[:3, :] = self.points.T

        self.points = (np.matmul(m, dummy))[:3, :].T
        self.update()

    def update(self):
        VTKMeshUtl.update_poly_w_points(self.mesh)
        self.cog = VTKMeshUtl.cog(self.mesh)
        self.volume = VTKMeshUtl.volume(self.mesh)
        self.mc = self.cog

    def center_mesh(self):
        trf = np.eye(4)
        trf[0:3, 3] = -self.cog
        self.apply_transform(trf)

    def translate_mesh(self, trl):
        trf = np.eye(4)
        trf[0:3, 3] = trl
        self.apply_transform(trf)

    def primary_axes(self):
        # primary X, secondary y, tertiary Z
        centered_data, cm = Stat.center_data(self.points)
        c = Stat.covariance_matrix(centered_data)
        e1, v1 = Stat.eig_rh(c)
        return v1, cm

    def rotate_points(self, r):
        # assumes r is 3x3
        new_points = np.matmul(r, self.points.transpose())
        return new_points.transpose()

    def principle_alignment(self, cogmcpstv=False):
        # center mesh to cloud centre
        v0, t0 = self.primary_axes()
        trf = np.eye(4)
        trf[0:3, 0:3] = v0.transpose()
        trf[:3, 3] = t0
        self.apply_transform(trf)
        diff = self.cog - self.mc
        if cogmcpstv and diff[0] < 0:
            # assumes cog and mesh centre is different
            r = Rotation.from_euler("xyz", [0, 0, np.pi])
            trf = np.eye(4)
            trf[0:3, 0:3] = r
            self.apply_transform(trf)
        pass

    def write_mesh(self, filename):
        VTKMeshUtl.write(filename, self.mesh)

class IMU(Yatsdo):
    '''
    This is IMU data storage object
    '''

    window_frame = 0
    window_time = 1

    def __init__(self, data: [pd.DataFrame, np.ndarray], col_names: list = None, acc: list = None, gyro: list = None,
                 mag:list = None, ori=None):
        if col_names is None:
            col_names = []
        super().__init__(data, col_names)
        self.acc_id: Optional[list] = acc
        self.gyr_id: Optional[list] = gyro
        self.mag_id: Optional[list] = mag
        self.ori = ori
        self.ori_filtered = True


    @property
    def acc_gyr_data(self):
        acc_gyr_id = []
        for i in self.acc_id:
            acc_gyr_id.append(i)
        for j in self.gyr_id:
            acc_gyr_id.append(j)
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.column_labels[acc_gyr_id]]
        elif isinstance(self.data, np.ndarray):
            ret = pd.DataFrame(data=self.data, columns=self.column_labels)
            get_ = [self.column_labels[c] for c in acc_gyr_id]
            get_.insert(0, "time")
            return ret[get_]

    @property
    def acc(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.column_labels[self.acc_id]]
        elif isinstance(self.data, np.ndarray):
            ret = pd.DataFrame(data=self.data, columns=self.column_labels)
            get_ = [self.column_labels[c] for c in self.acc_id]
            get_.insert(0, "time")
            return ret[get_]

    @property
    def gyr(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.column_labels[self.gyr_id]]
        elif isinstance(self.data, np.ndarray):
            ret = pd.DataFrame(data=self.data, columns=self.column_labels)
            get_ = [self.column_labels[c] for c in self.gyr_id]
            get_.insert(0, "time")
            return ret[get_]

    @property
    def mag(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.column_labels[self.mag_id]]
        elif isinstance(self.data, np.ndarray):
            ret = pd.DataFrame(data=self.data, columns=self.column_labels)
            get_ = [self.column_labels[c] for c in self.mag_id]
            get_.insert(0, "time")
            return ret[get_]

    @acc.setter
    def acc(self, data):
        raise NotImplementedError

    @gyr.setter
    def gyr(self, data):
        raise NotImplementedError

    @mag.setter
    def mag(self, data):
        raise NotImplementedError

    @staticmethod
    def create(s: StorageIO, key='TS'):
        rate = s.info['devices']['Device Sampling Rate']
        time = [i*(1/rate) for i in range(0, s.data.shape[0])]
        imu_labels = [c for c in s.data.columns if ('TS' in c and 'acc' in c) or ('TS' in c and 'gyr' in c) or ('TS' in c and 'mag' in c)]
        imu_id = []
        for c in imu_labels:
            elem = c.split('-')[1]
            idx = (key+'-'+elem).strip()
            if idx not in imu_id:
                imu_id.append(idx)
        imu_labels.insert(0, "time")
        imu_labels_dict = {"a": imu_labels}
        if len(imu_id) > 1:
            imu_labels_dict = {c: [d for d in imu_labels if c in d or 'time' in d] for c in imu_id}
        ret = {}

        def set_imu(imu_ls, s0, time0):
            temp = s0[imu_ls[1:]]
            data = np.zeros([temp.shape[0], temp.shape[1] + 1])
            data[:, 0] = time0
            data[:, 1:] = temp
            imu = IMU(data, imu_ls)
            imu.acc_id = [c for c in range(0, len(imu_labels)) if 'acc' in imu_labels[c].lower()]
            imu.gyr_id = [c for c in range(0, len(imu_labels)) if 'gyr' in imu_labels[c].lower()]
            imu.mag_id = [c for c in range(0, len(imu_labels)) if 'mag' in imu_labels[c].lower()]
            return imu

        for k in imu_labels_dict:
            ret[k] = set_imu(imu_labels_dict[k], s.data[imu_labels_dict[k][1:]], time)
        return ret

    def butterworth_filter(self, cut_off, sampling_rate):
        for d in range(1, self.acc.shape[1]):
            data = self.acc.iloc[:, d].to_numpy()
            filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
            self.data.iloc[:, self.acc_id[d-1]] = filtered
        
        for d in range(1, self.gyr.shape[1]):
            data = self.gyr.iloc[:, d].to_numpy()
            filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
            self.data.iloc[:, self.gyr_id[d-1]] = filtered

        if self.mag is not None:
            for d in range(1, self.mag.shape[1]):
                data = self.mag.iloc[:, d].to_numpy()
                filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
                self.data.iloc[:, self.mag_id[d-1]] = filtered

        if self.ori is not None and not self.ori_filtered:
            for d in range(1, self.ori.shape[1]):
                data = self.ori.iloc[:, d].to_numpy()
                filtered = Butterworth.butter_low_filter(data, cut_off, sampling_rate, order=4)
                # Assuming self.ori is also stored in self.data or similar? 
                # Wait, IMU init: self.ori = ori. It's stored separately or strictly passed in?
                # Line 517: self.ori = ori. 
                # If self.ori is a DataFrame, update it.
                self.ori.iloc[:, d] = filtered


    def export(self, filename, boo=[True, True, False, True]):
        imu = [self.acc, self.gyr, self.mag, self.ori]
        ret = np.zeros([self.size[0], 1+(np.sum(boo)*3)])
        columns = ['time']
        boos = ['acc', 'gyr', 'mag', 'ori']
        ret[:, 0] = self.acc.x
        col = 1
        for i in range(0, len(boos)):
            if boo[i]:
                columns.append(boos[i]+'_x')
                columns.append(boos[i]+'_y')
                columns.append(boos[i]+'_z')
                ret[:, col: col+3] = imu[i].data[:, 1:]
                col = col + 3

        p = pd.DataFrame(data=ret, columns=columns)
        p.to_csv(filename, index=False)

"""
Below are the helper class to provide backward compatibility 

"""

# class OSIMStorage(OSIMStorageV2):
#     def __init__(self, data, col_names=None, fill_data=False, filename="", header=None, ext=".sto"):
#         super().__init__(data, col_names, fill_data, filename, header, ext)


class Yatsdo(Yatsdo):
    def __init__(self, data, col_names=[], fill_data=False, time_col=0):
        super().__init__(data, col_names, fill_data, time_col)
        print("This class - {0} - has move to the core package".format(type(self).__name__))


class StorageIO(StorageIO):
    def __init__(self, store: pd.DataFrame = None, buffer_size: int = 10):
        super().__init__(store, buffer_size)


class MYXML(MYXML):
    # This is a simple xml reader/ writer
    def __init__(self, filename):
        super().__init__(filename)



class XROMMUtil(XROMMUtil):

    def __init__(self):
        super().__init__()


class Stat(Stat):
    def __init__(self):
        super().__init__()


class Bapple(Bapple):
    def __init__(self, x: Yatsdo = None, y: Yatsdo = None):
        """
        Version 2 of the Bapple (simplified):
        This Bapple just holds the data, export and import using npz instead of csv
        Hopefully this is faster
        :param x: Measurement/ Input data (i.e. IMU) (Type: pandas Dataframe)
        :param y: Measurement/ Input data (i.e. Target) (Type: pandas Dataframe)
       """
        super().__init__(x, y)


class JSONSUtl(JSONSUtl):
    def __init__(self):
        super().__init__()


class TRC(TRC):
    def __init__(self, data, col_names=[], filename="", fill_data=False):
        super().__init__(data, col_names, filename, fill_data)
