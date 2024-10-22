"""
Compound statistical shape (CS2) modelling framework project for 12 Labours - EP2/ Scaffolds

   Copyright [2023] [Ted Yeung]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Default python packages
import os
import importlib.util


def module_checker(modx=""):
    print("mesh.Tools - script dependency check: {0}".format(modx))
    spam_spec = importlib.util.find_spec(modx)
    found = spam_spec is not None
    if not found:
        print("\t-> {0} Not Available Installing {0}".format(modx))
        os.system('python -m pip install {0}'.format(modx))
        importlib.import_module(modx)
        spam_spec = importlib.util.find_spec(modx)
        found = spam_spec is not None
        if found:
            print("\t-> {0} Installed and Available".format(modx))
        else:
            print("\t-> Done .... Please Rerun Script to continue!")
    else:
        print("\t-> {0} Available".format(modx))


# Required packages for different functions
try:
    module_checker("vtk")
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
except ModuleNotFoundError:
    print("\t-> vtk not founded and could not be installed")
    pass

try:
    module_checker("numpy")
    import numpy as np
except ModuleNotFoundError:
    print("\t-> numpy not founded and could not be installed")
    pass

try:
    module_checker("pymeshlab")
    import pymeshlab as pm
except ModuleNotFoundError:
    print("\t-> pymeshlab not founded and could not be installed")
    pass


class Tools:
    @staticmethod
    def mesh2exf(mesh_filename: str, output_meshname: str, region_name: str = "", group_name: str = "external",
                 print_info=True):
        if print_info:
            print("Tools - mesh2exf: {0} -> {1}".format(mesh_filename, output_meshname))
        """
        This script reads in a mesh file and export as exf data point set
        Supported mesh type is stl, ply and obj
        :param mesh_filename: full path to the mesh file
        :param output_meshname: full path to the save location/ name of the file
        :param region_name: name of the region (optional)
        :param group_name: name of the group (optional), default is external
        :return: -
        """
        header = "EX Version: 2\n"
        header += "Region: /{0}\n".format(region_name)
        header += "!#nodeset datapoints\n"
        header += "Shape. Dimension=0\n"
        header += "#Fields=1\n"
        header += "1) data_coordinates, coordinate, rectangular cartesian, real, #Components=3\n"
        header += " x. #Values=1 (value)\n"
        header += " y. #Values=1 (value)\n"
        header += " z. #Values=1 (value)\n"
        nodes = ""
        footer = " Group name: {0}\n".format(group_name)
        footer += "!#nodeset datapoints\n"
        footer += "Shape. Dimension=0\n"
        footer += "#Fields=0\n"
        node_list = ""

        # Reading mesh data
        if mesh_filename.endswith(".obj"):
            reader = vtk.vtkOBJReader()
        elif mesh_filename.endswith(".ply"):
            reader = vtk.vtkPLYReader()
        elif mesh_filename.endswith(".stl"):
            reader = vtk.vtkSTLReader()
        else:
            print("Unknown file format")
            return

        reader.SetFileName(mesh_filename)
        reader.Update()

        # Extracting points data
        polydata = reader.GetOutput()
        points = np.asarray([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
        for i in range(0, points.shape[0]):
            node = "Node: {0}\n".format(i + 1)
            idx = str(i + 1)
            max_idx = str(points.shape[0])
            n = idx.rjust(len(max_idx), ' ')
            node_list += " Node: {0}\n".format(n)
            p = points[i, :]
            for j in range(0, 3):
                node += " {: .15e}\n".format(p[j])
            nodes += node
            pass

        # Writing out to exf file
        if len(output_meshname) > 0 and output_meshname.endswith(".exf"):
            try:
                with open(output_meshname, 'w') as f:
                    f.write(header)
                    f.write(nodes)
                    f.write(footer)
                    f.write(node_list)
            except IOError:
                print("error saving the exf file")
                pass
        else:
            print("Invalid output filename or path")

    @staticmethod
    def transform(matr: list = None, mesh_filename: str = "", out_file_path: str = "", print_info=False):
        if print_info:
            print("Tools - transform: {0} -> {1}".format(mesh_filename, out_file_path))
        """
        This function take a list of transformation matrices and apply them to the mesh and export the results.
        The order of the matrices in the list will be the order the transformation will be applied
        :param matr: A list of 4x4 affine transformation matrices
        :param mesh_filename: full path to the mesh file
        :param out_file_path: full path to the save location/ name of the file
        :return: -
        """
        if matr is None:
            return

        # Reading mesh file
        if mesh_filename.endswith(".obj"):
            reader = vtk.vtkOBJReader()
        elif mesh_filename.endswith(".ply"):
            reader = vtk.vtkPLYReader()
        elif mesh_filename.endswith(".stl"):
            reader = vtk.vtkSTLReader()
        else:
            print("Unknown file format")
            return
        reader.SetFileName(mesh_filename)
        reader.Update()

        # getting points
        polydata = reader.GetOutput()
        points = np.asarray([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])

        # applying rigid body transform (4x4 affine)
        x = np.ones([4, points.shape[0]])
        x[0:3, :] = points.transpose()
        for m in matr:
            x = np.matmul(m, x)

        # updating mesh data
        vs = vtk.vtkPoints()
        vs.SetData(numpy_to_vtk(x[0:3, :].transpose()))
        polydata.SetPoints(vs)

        # write out as obj
        w = vtk.vtkOBJWriter()
        w.SetInputData(polydata)
        w.SetFileName(out_file_path)
        w.Write()
        pass

    @staticmethod
    def flip(mesh: str, x=True, y=False, z=False):
        """
        Incomplete method - currently only flip in x direction
        :param mesh:
        :param x: set whether to flip x
        :param y: set whether to flip y
        :param z: set whether to flip z
        :return:
        """
        output_path = mesh[:mesh.rindex("\\")]
        fname = mesh.split("\\")
        if len(fname) < 2:
            output_path = mesh[:mesh.rindex("/")]
            fname = mesh.split("/")
        ms = pm.MeshSet()
        ms.load_new_mesh(mesh)
        if x:
            ms.apply_matrix_flip_or_swap_axis(flipx=True)
            ms.meshing_invert_face_orientation()
        if y:
            ms.apply_matrix_flip_or_swap_axis(flipy=True)
            ms.meshing_invert_face_orientation()
        if z:
            ms.apply_matrix_flip_or_swap_axis(flipz=True)
            ms.meshing_invert_face_orientation()
        if x or y or z:
            print(output_path + "\\" + fname[-1][:fname[-1].rindex(".")] + ".obj")
            ms.save_current_mesh(output_path + "\\" + fname[-1][:fname[-1].rindex(".")] + ".obj")


def mesh2exf_example():
    folder = "C:\\Users\\tyeu008\\Documents\\Repos\\ShapeSegTools\\src\\studies\\femur\\\samples\\"
    a = [b for b in os.listdir(folder) if b.endswith(".obj") and "left" in b and "transform" in b]
    for b in a:
        Tools.mesh2exf(folder + b, folder + b[:b.rindex(".")] + ".exf")


def transformation_example():
    folder = "C:\\Users\\tyeu008\\Documents\\Repos\\ShapeSegTools\\src\\studies\\femur\\\samples\\"
    a = [b for b in os.listdir(folder) if b.endswith(".obj") and "left" in b]
    translation = np.eye(4, 4)
    translation[0:3, 3] = [128, 194, -497]
    affine = np.eye(4, 4)
    affine[0, :] = [9.330117e-01, -3.552222e-01, -5.750060e-02, 6.000000e+00]
    affine[1, :] = [3.581496e-01, 9.321787e-01, 5.264688e-02, -7.000000e+00]
    affine[2, :] = [3.489950e-02, -6.971398e-02, 9.969564e-01, 1.200000e+01]
    tranform = [translation, affine]
    for b in a:
        Tools.transform(matr=tranform, mesh_filename=folder + b, out_file_path=folder + "transformed_" + b)


if __name__ == '__main__':
    run_transformation_example = False
    run_mesh2exf_example = True

    if run_transformation_example:
        transformation_example()
    if run_mesh2exf_example:
        mesh2exf_example()
