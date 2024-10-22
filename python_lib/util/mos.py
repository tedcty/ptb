import os
import pandas as pd
from subprocess import Popen, PIPE
from os import path

'''
This module contains methods dealing with os operations
i.e.
Getting a list of hard drive names etc
'''


def list_harddrives():
    """
    This function returns a table of hard drives info:
    'DeviceID', 'VolumeName', 'Description'

    Note: currently this is hardcoded
    :return: p, Panda Dataframe of hard drives in/connected to the PC
    """
    # Assumes fix column size
    wmic_path = "C:/Windows/System32/WMIC.exe"
    if not path.exists(wmic_path):
        wmic_path = "C:/Windows/System32/wbem/WMIC.exe"

    if path.exists(wmic_path):
        drives = os.popen(wmic_path+" logicaldisk get deviceid, volumename, description").readlines()
        lines = []
        boo = True
        di = 0
        vn = 0
        for d in drives:
            if len(d) > 1:
                lc = d[:-1].strip()
                if boo:
                    di = lc.index("DeviceID")
                    vn = lc.index("VolumeName")
                    boo = False

                la = [lc[:di].strip(), lc[di:vn].strip(), lc[vn:].strip()]
                lines.append(la)
        p = pd.DataFrame(data=lines[1:], columns=lines[0])
    elif path.exists('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'):
        print("Trying alternative powershell method ...")
        pipe = Popen(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', 'Get-Volume'], stdin=PIPE, stdout=PIPE,
                  stderr=PIPE)
        output, err = pipe.communicate()
        drives = output.decode("utf-8")
        drs = drives.strip().splitlines()
        columns_index = drs[1].split(' ')
        columns_size = [len(columns_index[i]) for i in range(len(columns_index))]
        columns_id = [0]
        current = 0
        for i in range(len(columns_size)):
            columns_id.append(columns_size[i] + current)
            current = current + columns_size[i] + 1
        col_names = [(drs[0][columns_id[i]:columns_id[i+1]]).strip() for i in range(len(columns_id)-1)]
        lines = []
        for j in range(2, len(drs)):
            lines.append([(drs[j][columns_id[i]:columns_id[i+1]]).strip() for i in range(len(columns_id)-1)])
        p = pd.DataFrame(data=lines, columns=col_names)
        p.rename(columns={'DriveLetter': 'DeviceID', 'FriendlyName': 'VolumeName'}, inplace=True)
    else:
        p = None
    if p is not None:
        col_new_order = ['DeviceID', 'VolumeName', 'Description']
        p = p.reindex(columns=col_new_order)
        print()
    return p


def find_drive_letter(volume_name):
    d = list_harddrives()
    ids = d[d['VolumeName'] == volume_name].index.values.astype(int)[0]
    d_letter = d["DeviceID"].iloc[ids]
    if ":" not in d_letter:
        d_letter = d_letter+":"
    return d_letter


if __name__ == '__main__':
    df = list_harddrives()
    pass

