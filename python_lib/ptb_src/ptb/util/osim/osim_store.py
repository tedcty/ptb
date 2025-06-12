import os
from enum import Enum
import csv

from ptb.core import Yatsdo
from ptb.util.io.helper import StorageIO, StorageType


class OSIMBoolean:
    def __init__(self, value):
        self.boo = value

    @staticmethod
    def convert(value):
        if isinstance(value, str):
            if 'yes' in value.lower():
                return True
            elif 'no' in value.lower():
                return False
        elif isinstance(value, bool):
            if value:
                return 'yes'
            else:
                return 'no'
        return None

class OSIMForcePlate(Enum):
    time = 0
    ground_force = ['ground_force', 'v', 'p']
    ground_torque = 'ground_torque'

    @staticmethod
    def map_from_c3d(m, p:int):
        if m == OSIMForcePlate.ground_force:
            return {'v': ['Force.Fx{0}'.format(p), 'Force.Fy{0}'.format(p), 'Force.Fz{0}'.format(p)],
                    'p': ['COP.Px{0}'.format(p), 'COP.Py{0}'.format(p),'COP.Pz{0}'.format(p)]}
        if m == OSIMForcePlate.ground_torque:
            return ['Moment.Mx{0}'.format(p), 'Moment.My{0}'.format(p), 'Moment.Mz{0}'.format(p)]
        return None

    def generate_label(self, plate:int):
        ret = {}
        xyz = ['x', 'y', 'z']
        if isinstance(self.value, list):
            for i in range(1, len(self.value)):
                v = []
                for j in range(0, 3):
                    s = "{0}{1}_{2}{3}".format(self.value[0], plate,self.value[i], xyz[j])
                    v.append(s)
                ret[self.value[i]] = v
        elif isinstance(self.value, str):
            v = []
            for j in range(0, 3):
                s = "{0}{1}_{2}".format(self.value, plate, xyz[j])
                v.append(s)
            ret['d'] = v
        else:
            ret['t'] = 'time'
        return ret


class HeadersLabels(Enum):
    trial = ["type", str]
    version = ["version", int]
    nRows = ["nRows", int]
    nColumns = ["nColumns", int]
    inDegrees = ["inDegrees", OSIMBoolean.convert]
    endheader = ["endheader", str]
    data_namme = ["name", str]
    notes = ["notes", str]

    @staticmethod
    def get(key_name):
        for header in HeadersLabels:
            if header.value[0] == key_name:
                return header
        return None


class OsimStorageV1(object):
    # Simple Storage Object that mimics sto files used with opensim
    def __init__(self, in_filenamme=None, out_filenamme=None):
        self.header = {HeadersLabels.nRows: 0,
                       HeadersLabels.nColumns: 0,
                       HeadersLabels.version: 1,
                       HeadersLabels.inDegrees: False}
        self.headings = []
        self.data = {}  # uses heading as keys
        if in_filenamme is not None:
            self.parse(in_filenamme)
            self.data_file = in_filenamme
            self.data_file_out = self.data_file[0:-4]+"copy.sto"
        if out_filenamme is not None:
            self.data_file_out = out_filenamme
            name_start = out_filenamme.rfind('/') + 1
            self.header[HeadersLabels.data_namme] = self.data_file_out[name_start:-4]
            if self.header == 0:
                self.header[HeadersLabels.nRows] = 0
                self.header[HeadersLabels.nColumns] = 0
                self.header[HeadersLabels.version] = 1
                self.header[HeadersLabels.inDegrees] = False

    def parse(self, filename):
        if filename is None:
            print("> Name Check: Filename is none ... stopping parser")
            return
        ik = []
        with open(filename, newline='') as csvfile:
            ik_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            i = 0
            data_start = False
            for a in ik_reader:
                if not (a[0] == HeadersLabels.endheader.value):
                    if "=" in a[0]:
                        parts = a[0].split("=")
                        if parts[0] == HeadersLabels.version.value:
                            self.header[HeadersLabels.version] = parts[1]
                        elif parts[0] == HeadersLabels.nRows.value:
                            self.header[HeadersLabels.nRows] = int(parts[1])
                        elif parts[0] == HeadersLabels.nColumns.value:
                            self.header[HeadersLabels.nColumns] = int(parts[1])
                        elif parts[0] == HeadersLabels.inDegrees.value:
                            yes_no = True
                            if parts[1] == "no":
                                yes_no = False
                            self.header[HeadersLabels.inDegrees] = yes_no
                    else:
                        if not data_start:
                            self.header[HeadersLabels.data_namme] = a[0]
                else:
                    data_start = True
                if data_start:
                    if a[0] == "time":
                        self.headings = a
                        for h in a:
                            self.data[h] = []
                    elif len(a) > 1:
                        i = 0
                        for h in self.headings:
                            self.data[h].append(float(a[i]))
                            i += 1
                        ik.append(a)
                i += 1

    def to_export(self):
        yes_no_degree = "no"
        if self.header[HeadersLabels.inDegrees]:
            yes_no_degree = "yes"
        buffer = [[self.header[HeadersLabels.data_namme]],
                  [HeadersLabels.version.value + "=" + str(self.header[HeadersLabels.version])],
                  [HeadersLabels.nRows.value + "=" + str(self.header[HeadersLabels.nRows])],
                  [HeadersLabels.nColumns.value + "=" + str(self.header[HeadersLabels.nColumns])],
                  [HeadersLabels.inDegrees.value + "=" + yes_no_degree],
                  [HeadersLabels.endheader.value],
                  self.headings]
        try:
            for i in range(0, int(self.header[HeadersLabels.nRows])):
                row = []
                for h in self.headings:
                    row.append(self.data[h][i])
                buffer.append(row)
        except IndexError:
            pass
        return buffer

    def search_headings(self, tar):
        ids = []
        for i in range(0, len(self.headings)):
            if tar in self.headings[i]:
                ids.append(i)
        return ids

    def get(self, heading_id, rows=-1):
        a = []
        if rows == -1:
            for i in heading_id:
                a.append(self.data[self.headings[i]])
        return a

    def write(self):
        out_buffer = self.to_export()
        try_count = 0
        try:
            with open(self.data_file_out, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter='\t',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for s in out_buffer:
                    spamwriter.writerow(s)
        except PermissionError:
            print("Encountered permission error while writing data to " + self.data_file_out)
            self.data_file_out = self.data_file_out[0:-4] + "_temp." + self.data_file_out[-3:]
            print("|-->Will try to write to a temp file: " + self.data_file_out)
            if try_count < 100:
                self.write()
                try_count += 1
            else:
                print("Name too long: " + self.data_file_out)


class OSIMStorageV2(Yatsdo):
    """
    Example of header:

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
    def read(filename: str):
        """
        This method reads mot/sto opensim files and create a OSIMStorageV2
        :param filename: file path
        :return: An instance of OSIMStorageV2 Object
        """
        if not os.path.exists(filename):
            return None
        s = StorageIO.load(filename, StorageType.mot)

        k = s.info["header"]
        header = {HeadersLabels.trial: k[0].strip()}

        for i in range(1, len(k)):
            if '=' in k[i]:
                parts = k[i].split('=')
                h = HeadersLabels.get(parts[0].strip())
                typ = h.value[1]
                attr = typ(parts[1].strip())
                header[h] = attr
            else:
                h = HeadersLabels.get(k[i].strip())
                typ = h.value[1]
                header[h] = typ(k[i].strip())

        ret = OSIMStorageV2(s.data, filename=filename, header=header)
        return ret

    def update(self):
        super().update()
        self.header[HeadersLabels.nRows] = self.data.shape[0]
        self.header[HeadersLabels.nColumns] = self.data.shape[1]
        pass

    def header2string(self, h):
        if h not in [HeadersLabels.trial, HeadersLabels.notes, HeadersLabels.endheader]:
            if h not in [HeadersLabels.inDegrees]:
                return "{0}={1}\n".format(h.value[0], self.header[h])
            else:
                return "{0}={1}\n".format(h.value[0], OSIMBoolean.convert(self.header[h]))
        else:
            return "{0}{1}\n".format('', self.header[h])

    def write(self, filename):
        self.update()
        lines = [self.header2string(h) for h in self.header]
        cols = ""
        for c in self.col_labels:
            cols += c
            cols += "\t"
        lines.append(cols.strip() + "\n")
        for i in range(0, self.data.shape[0]):
            ret = ""
            for j in range(0, self.data.shape[1]):
                ret += "{0:.6f}\t".format(self.data[i,j])
            ret.strip()
            ret += '\n'
            lines.append(ret)
            pass

        with open(filename, 'w') as writer:
            writer.writelines(lines)
        pass

class OSIMStorage:

    @staticmethod
    def simple_header_template():
        header = {HeadersLabels.trial: 'default',
                  HeadersLabels.version: 1,
                  HeadersLabels.nRows: 0,
                  HeadersLabels.nColumns: 0,
                  HeadersLabels.inDegrees: False,
                  HeadersLabels.endheader: HeadersLabels.endheader.value[0]}
        return header

    @staticmethod
    def create(data, header, filename, store_version=OSIMStorageV2):
        """
        This method reads mot/sto opensim files and create an opensim storage object.
        Current creates an OSIMStorageV2 as the store but change be changed.
        :param filename: file path
        :param header: dictionary of header information
        :param data: panda Dataframe
        :param store_version: Storage class handler
        :return: An instance of opensim storage object
        """
        ret = store_version(data, filename=filename, header=header)
        ext = filename.split(".")[-1]
        if 'mot' in ext or 'sto' in ext:
            ret.ext = ".{0}".format(ext)
        return OSIMStorage(ret)

    @staticmethod
    def read(f, store_version=OSIMStorageV2):
        """
        This method reads mot/sto opensim files and create an opensim storage object.
        Current creates an OSIMStorageV2 as the store but change be changed.
        :param f: file path
        :param store_version: Storage version
        :return: An instance of opensim storage object
        """

        return OSIMStorage(store_version.read(f))

    def write(self, f):
        self.store.write(f)

    def __init__(self, store=None, ext='mot'):
        self.store = store
        self.ext = ext

# # Uncomment for testing
# if __name__ == "__main__":
#     # d = './walk20.mot'
#     # w = './walk20a.mot'
#     # osim_mot = OSIMStorage.read(d)
#     # osim_mot.write(w)
#     pass
