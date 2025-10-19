import os
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import vtk

from tqdm import tqdm
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

class VTKMeshUtl(Enum):
    ply = [".ply", vtk.vtkPLYReader, vtk.vtkPLYWriter]
    stl = [".stl", vtk.vtkSTLReader, vtk.vtkSTLWriter]
    obj = [".obj", vtk.vtkOBJReader, vtk.vtkOBJWriter]
    vtp = [".vtp", vtk.vtkXMLPolyDataReader, vtk.vtkXMLPolyDataWriter]
    none = [None, None, None]
    unknown = ["unknown", None, None]

    @staticmethod
    def color_convert(color):
        return [color[0] / 255.0, color[1] / 255, color[2] / 255]

    @staticmethod
    def smooth_mesh(polydata, iter=15, factor=0.2):
        # Create a vtkSmoothPolyDataFilter
        smooth_filter = vtk.vtkSmoothPolyDataFilter()
        smooth_filter.SetInputData(polydata)
        smooth_filter.SetNumberOfIterations(iter)  # adjust this value to increase smoothing
        smooth_filter.SetRelaxationFactor(factor)  # adjust this value to increase smoothing
        smooth_filter.FeatureEdgeSmoothingOff()
        smooth_filter.BoundarySmoothingOff()
        smooth_filter.Update()
        return smooth_filter.GetOutput()

    @staticmethod
    def merge_meshes(polys):
        appendFilter = vtk.vtkAppendPolyData()
        for p in polys:
            appendFilter.AddInputData(p)
        # Merge the meshes
        appendFilter.Update()
        return appendFilter.GetOutput()

    @staticmethod
    def point_cloud_to_ply(file_name, cols=None, transform=None):
        if cols is None:
            cols = ['X', 'Y', 'Z']
        # Load the point cloud data from a CSV file
        df = pd.read_csv(file_name)
        if transform is not None and isinstance(transform, np.ndarray):
            n = df.to_numpy()
            np_points = np.ones([4, n.shape[0]])
            np_points[:3, :] = n.T
            transformed = np.matmul(transform, np_points)
            df = pd.DataFrame(data=transformed[:3, :].T, columns=cols)

        # Create a vtkPoints object and insert the points into it
        points = vtk.vtkPoints()
        for index, row in df.iterrows():
            points.InsertNextPoint(row[cols[0]], row[cols[1]], row[cols[2]])

        # Create a vtkPolyData object and add the points to it
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        # Write the polydata to a .ply file
        plyWriter = vtk.vtkPLYWriter()
        plyWriter.SetFileName('{0}.ply'.format(file_name[:file_name.rindex('.')]))
        plyWriter.SetInputData(polydata)
        plyWriter.Write()

    @staticmethod
    def clone_poly(poly):
        p = vtk.vtkPolyData()
        p.DeepCopy(poly)
        return p

    @staticmethod
    def is_valid(filename):
        k = VTKMeshUtl.get_type(filename)
        if VTKMeshUtl.none == k or VTKMeshUtl.unknown:
            return False
        return True

    # @staticmethod
    # def test_load_mesh_as_actor():
    #     vertices = VTKMeshUtl.extract_points(self.mean_mesh)
    #     vertices0 = np.asarray([self.mean_mesh.GetPoint(i) for i in range(self.mean_mesh.GetNumberOfPoints())])

    @staticmethod
    def get_type(filename):
        if filename is None:
            return VTKMeshUtl.none
        if filename.endswith(".ply"):
            return VTKMeshUtl.ply
        if filename.endswith(".stl"):
            return VTKMeshUtl.stl
        if filename.endswith(".obj"):
            return VTKMeshUtl.obj
        if filename.endswith(".vtp"):
            return VTKMeshUtl.vtp
        return VTKMeshUtl.unknown

    @staticmethod
    def is_none(mesh_type):
        if VTKMeshUtl.none == mesh_type:
            return True
        else:
            return False

    def reader(self):
        return self.value[1]()

    def writer(self):
        try:
            w = self.value[2]()
        except TypeError:
            w.SetDataModeToAscii()
        return w

    def label(self):
        return self.value[0]

    @staticmethod
    def volume(polydata):
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputData(polydata)
        mass_props.Update()
        return mass_props.GetVolume()

    @staticmethod
    def center_of_mass(polydata):
        center_of_mass = vtk.vtkCenterOfMass()
        center_of_mass.SetInputData(polydata)
        center_of_mass.Update()
        return center_of_mass.GetCenter()

    @staticmethod
    def cog(polydata):
        return VTKMeshUtl.center_of_mass(polydata)

    @staticmethod
    def get_polydata_f_actor(actor):
        return actor.GetMapper().GetInput()

    @staticmethod
    def mesh2exf(mesh_filename: str, output_meshname: str, region_name: str = "", group_name: str = "external",
                 print_info=True):
        if print_info:
            print("mesh2exf: {0} -> {1}".format(mesh_filename, output_meshname))
        """
        This script reads in a mesh file and export as exf data point set.
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
    def slice_mesh_to_images(polydata, list_of_locations: list, pixel_spacing, slice_thickness=1.5,
                             image_size=(512, 512)):
        bar = tqdm(range(len(list_of_locations)),
                   desc="Making masks",
                   ascii=False, ncols=100, colour="#6e5b5b")
        ret = []
        for p in list_of_locations:
            blackImage = vtk.vtkImageData()
            blackImage.SetSpacing(pixel_spacing[0], pixel_spacing[1], slice_thickness)
            blackImage.SetDimensions(image_size[0], image_size[1], 1)
            blackImage.SetOrigin(p[1][0], p[1][1], p[1][2])
            blackImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            inval = 255
            outval = 0
            count = blackImage.GetNumberOfPoints()
            for i in range(0, count):
                blackImage.GetPointData().GetScalars().SetTuple1(i, inval)
            pol2stenc = vtk.vtkPolyDataToImageStencil()
            pol2stenc.SetInputData(polydata)
            pol2stenc.SetOutputOrigin(p[1][0], p[1][1], p[1][2])
            pol2stenc.SetOutputSpacing(pixel_spacing[0], pixel_spacing[1], 1.5)
            pol2stenc.SetOutputWholeExtent(blackImage.GetExtent())
            pol2stenc.Update()

            imgstenc = vtk.vtkImageStencil()
            imgstenc.SetInputData(blackImage)
            imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
            imgstenc.ReverseStencilOff()
            imgstenc.SetBackgroundValue(outval)
            imgstenc.Update()

            flip = vtk.vtkImageFlip()
            flip.SetInputConnection(imgstenc.GetOutputPort())
            flip.SetFilteredAxis(1)
            flip.FlipAboutOriginOn()
            flip.ReleaseDataFlagOn()
            flip.Update()
            ret.append(flip.GetOutput())
            bar.update(1)
        bar.close()
        return ret

    @staticmethod
    def sub_mesh(node_id_list, polydata, debug = False, try_speed_up=2):
        mesh_data_copy = vtk.vtkPolyData()
        mesh_data_copy.DeepCopy(polydata)
        cell_data = mesh_data_copy.GetPolys()
        # Get the point data and cell data of the mesh
        if debug:
            # currently slow for large meshes
            points_l = []
            if try_speed_up == 1:
                points_l = [mesh_data_copy.GetPoint(node_id_list[i]) for i in range(len(node_id_list))]
            else:
                for i in range(len(node_id_list)):
                    p = mesh_data_copy.GetPoint(node_id_list[i])
                    points_l.append(p)

        if try_speed_up == 1:
            def steps(i):
                cell = mesh_data_copy.GetCell(i)
                p1 = cell.GetPointId(0)
                p2 = cell.GetPointId(1)
                p3 = cell.GetPointId(2)
                if not (p1 in node_id_list and p2 in node_id_list and p3 in node_id_list):
                    mesh_data_copy.DeleteCell(i)
                return 1
            temp = [steps(i) for i in range(cell_data.GetNumberOfCells())]

        else:
            for i in range(cell_data.GetNumberOfCells()):
                cell = mesh_data_copy.GetCell(i)
                p1 = cell.GetPointId(0)
                p2 = cell.GetPointId(1)
                p3 = cell.GetPointId(2)
                if not (p1 in node_id_list and p2 in node_id_list and p3 in node_id_list):
                    mesh_data_copy.DeleteCell(i)
        mesh_data_copy.RemoveDeletedCells()
        return VTKMeshUtl.clean(mesh_data_copy)

    @staticmethod
    def closest_point(tar_point, polydata, idx=-1):
        octree = vtk.vtkOctreePointLocator()
        octree.SetDataSet(polydata)
        octree.BuildLocator()
        result = octree.FindClosestPoint(tar_point)
        point = polydata.GetPoint(result)
        diff = np.linalg.norm(point - tar_point)
        return [idx, result, diff]

    @staticmethod
    def closest_point_set(point_set, polydata):
        arg_list = [[point_set[i, :], polydata, i] for i in range(0, point_set.shape[0])]
        ret = [VTKMeshUtl.closest_point(w[0], w[1], w[2]) for w in arg_list]
        return pd.DataFrame(data=ret, columns=["idx", "idm", "errors"])

    @staticmethod
    def apply_icp(__source__, icp):
        icp_transform_filter = vtk.vtkTransformPolyDataFilter()
        icp_transform_filter.SetInputData(__source__)
        icp_transform_filter.SetTransform(icp)
        icp_transform_filter.Update()
        return icp_transform_filter.GetOutput()

    @staticmethod
    def icp(__source__, __target__, num_iter=1000):
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(__source__)
        icp.SetTarget(__target__)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMeanDistanceModeToRMS()
        icp.SetMaximumMeanDistance(1e-6)
        # icp.DebugOn()
        icp.SetMaximumNumberOfIterations(num_iter)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        return icp

    @staticmethod
    def mirror(_polydata_: vtk.vtkPolyData):
        m1 = np.eye(4)
        m1[1, 1] = -1
        m2 = np.eye(4)
        m2[0, 0] = -1
        m2[1, 1] = -1
        m3 = np.matmul(m2, m1)
        vtk_points: np.ndarray = VTKMeshUtl.extract_points(_polydata_)
        points = np.ones([4, vtk_points.shape[0]])
        points[:3, :] = vtk_points.T
        new_points = np.matmul(m3, points)[:3, :].T
        return VTKMeshUtl.update_poly_w_points(new_points, _polydata_)

    @staticmethod
    def draw_line_between_two_points(orig, p1, colour=None, line_width=5):
        if colour is None:
            colors = vtk.vtkNamedColors()
            v = colors.GetColor3d('Tomato')
            colour = [v.GetRed(), v.GetGreen(), v.GetBlue()]
        if isinstance(colour, str):
            colors = vtk.vtkNamedColors()
            v = colors.GetColor3d(colour)
            colour = [v.GetRed(), v.GetGreen(), v.GetBlue()]

        points = vtk.vtkPoints()
        points.InsertNextPoint(orig)
        points.InsertNextPoint(p1)

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(2)
        for i in range(0, 2):
            polyLine.GetPointIds().SetId(i, i)

        # Create a cell array to store the lines in and add the lines to it
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)

        # Create a polydata to store everything in
        polyData = vtk.vtkPolyData()

        # Add the points to the dataset
        polyData.SetPoints(points)

        # Add the lines to the dataset
        polyData.SetLines(cells)

        # Setup actor and mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colour)
        actor.GetProperty().SetLineWidth(line_width)
        return actor

    @staticmethod
    def clip_and_fill(poly, origin, norm=None, radius=1, cut_type='plane'):
        cut_obj = None
        if cut_type == 'plane' and norm is not None:
            plane = vtk.vtkPlane()
            plane.SetOrigin(origin[0], origin[1], origin[2])
            plane.SetNormal(norm[0], norm[1], norm[2])
            cut_obj = plane
        elif cut_type == 'sphere':
            sphere = vtk.vtkSphere()
            sphere.SetCenter(origin[0], origin[1], origin[2])
            sphere.SetRadius(radius)
            pass
        clipper = vtk.vtkClipPolyData()
        clipper.SetClipFunction(cut_obj)
        clipper.SetInputConnection(poly)
        clipper.Update()

        # Use vtkFillHolesFilter to fill the hole created by clipping
        fillHoles = vtk.vtkFillHolesFilter()
        fillHoles.SetInputConnection(clipper.GetOutputPort())
        fillHoles.SetHoleSize(1000.0)  # Adjust the hole size limit as needed
        fillHoles.Update()

        filledMapper = vtk.vtkPolyDataMapper()
        filledMapper.SetInputConnection(fillHoles.GetOutputPort())
        return filledMapper


    @staticmethod
    def cal_normals(poly):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.ComputeCellNormalsOn()
        normals.SetAutoOrientNormals(True)
        normals.Update()
        return normals

    @staticmethod
    def normals_as_numpy(normals: vtk.vtkPolyDataNormals):
        output = normals.GetOutput()
        cell_data = output.GetPointData()
        _normals = cell_data.GetNormals()
        return np.array([_normals.GetTuple3(f) for f in range(_normals.GetNumberOfTuples())])

    @staticmethod
    def extract_points(_data_: Union[vtk.vtkPolyData, vtk.vtkActor]):
        if isinstance(_data_, vtk.vtkPolyData):
            return vtk_to_numpy(_data_.GetPoints().GetData())
        return vtk_to_numpy(_data_.GetMapper().GetInput().GetPoints().GetData())


    @staticmethod
    def make_sphere(loc, size=3.0, colour=None):
        if colour == None:
            colour = "Cornsilk"
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(loc[0], loc[1], loc[2])
        sphereSource.SetRadius(size)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d(colour))
        return actor, sphereSource

    @staticmethod
    def update_poly_w_points(_points_: np.ndarray, _polydata_: vtk.vtkPolyData):
        _vh_ = vtk.vtkPoints()
        _vh_.SetData(numpy_to_vtk(_points_))
        _polydata_.SetPoints(_vh_)
        return _polydata_

    @staticmethod
    def extract_triangles(_polydata_: vtk.vtkPolyData):
        # assumes that faces are triangular
        poly_data = _polydata_.GetPolys().GetData()
        face_indices = vtk_to_numpy(poly_data)

        face_indices = face_indices.reshape((-1, 4))
        _nFaces = face_indices.shape[0]
        _triangles = face_indices[:, 1:].copy()
        return {"n_faces": _nFaces, "triangles": _triangles}

    @staticmethod
    def clean(polydata):
        clean_poly = vtk.vtkCleanPolyData()
        clean_poly.SetInputData(polydata)
        clean_poly.Update()
        return clean_poly.GetOutput()

    @staticmethod
    def write(filename, polydata):
        """

        :param filename: path
        :param polydata: polydata
        :return:
        """
        try:
            v = VTKMeshUtl.get_type(filename)
            cleaned = VTKMeshUtl.clean(polydata)
            w = v.writer()
            w.SetInputData(cleaned)
            w.SetFileName(filename)
            w.Write()
        except:
            print("Cannot output "+filename)

    def read(self, mesh_filename):
        if os.path.exists(mesh_filename):
            read = self.reader()
            read.SetFileName(mesh_filename)
            read.Update()
            return read.GetOutput()
        return None

    @staticmethod
    def load(mesh_filename, clean_mesh=True):
        """
        Load mesh as polydata
        :param mesh_filename: filepath of the mesh
        :param clean_mesh: remove unreferenced nodes
        :return: vtk.vtkPolyData
        """
        if os.path.exists(mesh_filename):
            mesh_type = VTKMeshUtl.get_type(mesh_filename)
            poly = mesh_type.read(mesh_filename)
            if clean_mesh and poly is not None:
                return VTKMeshUtl.clean(poly)
            else:
                return poly
        else:
            print("File not found: {0} does not exist!".format(mesh_filename))
            return None


class XROMMUtil:
    @staticmethod
    def save_maya_cam(output_filename, image_size: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray, dist=None):
        image = 'image size\n{0:d}, {1:d}\n\n'.format(image_size[0], image_size[0])
        camera = 'camera matrix\n{0:f},{0:f},{0:f}\n{0:f},{0:f},{0:f}\n{0:f},{0:f},{0:f}\n\n'.format(
            k[0, 0], k[0, 1], k[0, 2], k[1, 0], k[1, 1], k[1, 2], k[2, 0], k[2, 1], k[2, 2]
        )
        rotation = 'rotation\n{0:f},{0:f},{0:f}\n{0:f},{0:f},{0:f}\n{0:f},{0:f},{0:f}\n\n'.format(
            r[0, 0], r[0, 1], r[0, 2], r[1, 0], r[1, 1], r[1, 2], r[2, 0], r[2, 1], r[2, 2]
        )
        translation = 'translation\n{0:f}\n{0:f}\n{0:f}\n\n'.format(
            t[0], t[1], t[2]
        )
        if dist is not None:
            print("Not Implemented!")
        try:
            file1 = open(output_filename, "w")  # write mode
            file1.write(image+camera+rotation+translation)
            file1.close()
        except IOError:
            print("Write Error!")

