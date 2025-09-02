# ptb a MMG Repository of useful tools
This is a repository containing useful tools and development of useful tools for analysing, processing and modelling in mostly python (but also matlab)

**Warning: Matlab code is old and no longer being maintained.**

Contributors: Ted Yeung, Isabella Murrell, Homayoon Zarshenas, Thorben Pauli.

## Requirements
Opensim (tested on Opensim 4.5):
conda install -y -c opensim-org opensim

Python 3.10 or above

Packages: 'pandas', 'scipy', 'numpy', 'scikit-learn', 'Pillow', 'vtk', 'PySide6', 'tsfresh'

## Supported Data formats
### Mocap
c3d, trc, mot (Opensim), sto(Opensim), csv

### Meshes
OBJ, STL, Ply, Vtk

Setup
For using the PTB: * It can be set up using the wheel in the dist folder, which will install ptb into your Python environment.
