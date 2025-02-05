# coding=utf-8
import pandas as pd
import numpy as np
import vtk
import csv
from enum import Enum


class Xlsx_Reader(object):

    @staticmethod
    def read(filename):
        xl = pd.ExcelFile(filename)
        print(xl.sheet_names)
        df1 = xl.parse(xl.sheet_names[0])
        for d in df1:
            if d == "Values":
                data = df1.get(d)
                print(data)


class TRC_Header(Enum):
    PathFileType = "PathFileType"
    DataRate = "DataRate"
    CameraRate = "CameraRate"
    NumFrames = "NumFrames"
    NumMarkers = "NumMarkers"
    Units = "Units"
    OrigDataRate = "OrigDataRate"
    OrigDataStartFrame = "OrigDataStartFrame"
    OrigNumFrames = "OrigNumFrames"
    Euler = "Euler"
    Filename = "Filename"
    Frame_tag = "Frame#"
    Time = "Time"


class TRC(object):
    # Simple TRC reader
    def __init__(self):
        self.header = {}
        self.labels = {}
        self.data = {}

    @staticmethod
    def read(filename):
        t = TRC()
        with open(filename) as f:
            read_data = f.read()
        f.closed
        lines = read_data.split("\n")
        line_num = 0
        for l in lines:
            sl = l.split("\t")
            if len(sl) == 1:
                sl = l.split(",")
            if line_num == 0:

                t.header[TRC_Header.PathFileType] = float(sl[1].strip())
                t.header[TRC_Header.Euler] = sl[2].strip()
                t.header[TRC_Header.Filename] = sl[3].strip()
            elif line_num == 2:
                t.header[TRC_Header.DataRate] = float(sl[0].strip())
                t.header[TRC_Header.CameraRate] = float(sl[1].strip())
                t.header[TRC_Header.NumFrames] = float(sl[2].strip())
                t.header[TRC_Header.NumMarkers] = float(sl[3].strip())
                t.header[TRC_Header.Units] = sl[4].strip()
                t.header[TRC_Header.OrigDataRate] = float(sl[5].strip())
                t.header[TRC_Header.OrigDataStartFrame] = float(sl[6].strip())
                t.header[TRC_Header.OrigNumFrames] = float(sl[7].strip())
            elif line_num == 3:
                t.labels[TRC_Header.Frame_tag.value] = sl[0]
                t.labels[TRC_Header.Time.value] = sl[1]
                marker_count = 1
                for s in sl[2:-1]:
                    print(s.split(":")[0])

            line_num += 1

        for h in t.header:
            print(h.value +" "+str(t.header[h])+"")


class MeshIO(object):

    @staticmethod
    def stl_2_ply(in_file, out_file):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(in_file)
        reader.Update()
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(reader.GetOutput())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())
        mapper.Update()
        polydata = vtk.vtkPolyData()
        polydata.DeepCopy(mapper.GetInput())
        print(polydata)

        plyWriter = vtk.vtkPLYWriter()
        plyWriter.SetFileName(out_file)
        plyWriter.SetInputConnection(reader.GetOutputPort())
        plyWriter.Write()


class csv_IO(object):

    def __init__(self, filename=None):
        self.filename = filename
        self.out_buffer = []
        self.try_count = 0

    def write(self):
        try:
            with open(self.filename, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for s in self.out_buffer:
                    spamwriter.writerow(s)
        except PermissionError:
            self.filename = self.filename[0:-4] + "_temp." + self.filename[-3:]
            if self.try_count < 100:
                self.write()
                self.try_count += 1
            print(self.filename)

    def read(self, astype=None, delim: str = ',', skip_row: int = 0, to_numpy: bool = True):
        ret = []
        head = 'ï»¿'
        header = []
        count = 0
        if self.filename is not None:
            with open(self.filename, newline='') as csvfile:
                data = csv.reader(csvfile, delimiter=delim, quotechar='|')
                testlen=-1
                for a in data:
                    if count >= skip_row:
                        if astype is "float":
                            if isinstance(a, list):
                                c = []
                                for b in a:
                                    try:
                                        d = float(b)
                                        c.append(d)
                                    except:
                                        continue
                                if len(c) >= testlen:
                                    testlen = len(c)
                                    ret.append(np.array(c))
                        else:
                            ret.append(a)
                    count = count + 1
                    #print(a)
        if to_numpy:
            numpret = np.zeros((len(ret), len(ret[0])))
            erroridex = [0, 0]
            try:
                for i in range(0, len(ret)):
                    erroridex[0] = i
                    for j in range(0, len(ret[0])):
                        erroridex[1] = j
                        numpret[i, j] = ret[i][j]
                return numpret
            except IndexError:
                print(erroridex)
                pass

        return ret
