import os

if __name__ == '__main__':
    print("Getting Requirement:")
    os.system('conda install -c opensim-org opensim')
    os.system('python -m pip install pandas scipy numpy==1.26.4 PySide6 tsfresh vtk')
    # for building the package
    os.system('python -m pip install build')
    print("Done!")
