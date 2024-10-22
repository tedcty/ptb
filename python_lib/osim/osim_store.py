from enum import Enum
import csv


class HeadersLabels(Enum):
    version = "version"
    nRows = "nRows"
    nColumns = "nColumns"
    inDegrees = "inDegrees"
    endheader = "endheader"
    data_namme = "name"
    notes = "notes"


class OsimStorage(object):
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


