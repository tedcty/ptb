import os

if __name__ == '__main__':
    print("Getting Requirement:")
    os.system('conda install -c opensim-org opensim')
    os.system('python -m pip install pandas scipy numpy vtk PySide6 tsfresh')
    # for building the package
    os.system('python -m pip install build')
    print("Done!")
