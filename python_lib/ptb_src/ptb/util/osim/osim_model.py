# coding=utf-8
from enum import Enum
import xml.sax
from xml.dom.minidom import parse, Node
import xml.dom.minidom
import numpy as np
import csv


"""
OsimLezer Reads in the Osim file and create a OsimModel obj

"""


class OsimTags(Enum):
    OpenSimDocument = "OpenSimDocument"
    Model = "Model"
    mass = "mass"
    mass_center = "mass_center"
    inertia_xx = "inertia_xx"
    inertia_yy = "inertia_yy"
    inertia_zz = "inertia_zz"
    inertia_xy = "inertia_xy"
    inertia_xz = "inertia_xz"
    inertia_yz = "inertia_yz"
    credits = "credits"
    publications = "publications"
    length_units = "length_units"
    force_units = "force_units"
    gravity = "gravity"
    body_set = "BodySet"
    constraint_set = "ConstraintSet"
    force_set = "ForceSet"
    marker_set = "MarkerSet"
    contact_geometry_set = "ContactGeometrySet"
    controller_set = "ControllerSet"
    probeSet = "ProbeSet"
    component_set = "ComponentSet"
    body = "Body"
    joint = "Joint"
    objects = "objects"
    parent_body = "parent_body"
    location_in_parent = "location_in_parent"
    orientation_in_parent = "orientation_in_parent"
    location = "location"
    orientation = "orientation"
    CoordinateSet = "CoordinateSet"
    motion_type = "motion_type"
    default_value = "default_value"
    default_speed_value = "default_speed_value"
    range = "range"
    clamped = "clamped"
    locked = "locked"
    prescribed_function = "prescribed_function"
    prescribed = "prescribed"
    SpatialTransform = "SpatialTransform"
    reverse = "reverse"
    TransformAxis = "TransformAxis"
    coordinates = "coordinates"
    axis = "axis"
    Marker = "Marker"
    fixed = "fixed"

    @staticmethod
    def is_tag(item):
        for e in OsimTags:
            if e.value == item:
                return True
        return False

    @staticmethod
    def get_tag(item):
        for e in OsimTags:
            if e.value == item:
                return e
        return None


class OsimAttributes(Enum):
    name = "name"
    version = "Version"


class OsimTypes(Enum):
    rotational = "rotational"
    translational = "translational"
    max = "max"
    min = "min"
    radian = "radian"
    degree = "degree"
    rotation1 = "rotation1"
    rotation2 = "rotation2"
    rotation3 = "rotation3"
    translation1 = "translation1"
    translation2 = "translation2"
    translation3 = "translation3"
    x = "x"
    y = "y"
    z = "z"
    linear = "linear"

    @staticmethod
    def get_type(item):
        for e in OsimTypes:
            if e.value == item:
                return e
        return None

    @staticmethod
    def bool(item):
        item = item.lower()
        if item == "false":
            return False
        elif item == "true":
            return True

    @staticmethod
    def axis(item):
        if item[0] == 1:
            return OsimTypes.x
        elif item[1] == 1:
            return OsimTypes.y
        elif item[2] == 1:
            return OsimTypes.z


class OsimMarker(object):
    def __init__(self):
        self.root_model = None
        self.name = "marker"
        self.body = None
        self.location_in_body = None
        self.fixed = False

    def parse(self, m):
        if m.hasAttribute(OsimAttributes.name.value):
            self.name = m.getAttribute(OsimAttributes.name.value)

        data = m.getElementsByTagName(OsimTags.body.value.lower())
        for d in data:
            target = d.childNodes[0].data.strip()
            self.body = self.root_model.find_body(target)

        data = m.getElementsByTagName(OsimTags.location.value.lower())
        for d in data:
            data_ax = d.childNodes[0].data.strip().split(" ")
            self.location_in_body = np.array([float(data_ax[0]), float(data_ax[1]), float(data_ax[2])])

        data = m.getElementsByTagName(OsimTags.fixed.value.lower())
        for d in data:
            target = d.childNodes[0].data.strip()
            self.fixed = OsimTypes.bool(target)


class OsimTransform(object):
    def __init__(self):
        self.type = OsimTags.SpatialTransform
        self.rotation_transforms = {}
        self.translation_transforms = {}


class OsimTransformAxis(object):
    def __init__(self):
        self.name = ""
        self.coordinate = None
        self.axis = None
        self.axis_function = OsimTypes.linear   # marcador de posicion no utilizado en el marco de referencia


class OsimJoint(object):
    def __init__(self):
        self.root_model = None
        self.name = ""
        self.joint_type = ""
        self.transform = None
        self.parent_body = None
        self.location_in_parent = None
        self.orientation_in_parent = None
        self.location = None
        self.orientation = None
        self.coordinate_set = {}
        self.reverse = False

    def __str__(self):
        return self.name + " / " + self.parent_body.name

    def parse(self, info, tag):
        if tag == OsimTags.parent_body:
            body_name = info.childNodes[0].data
            self.parent_body = self.root_model.find_body(body_name)
        elif tag == OsimTags.location_in_parent:
            data_ax = info.childNodes[0].data.strip().split(" ")
            filter_data_ax = []
            for d in data_ax:
                if len(d) > 0:
                    filter_data_ax.append(d)
            self.location_in_parent = np.array([float(filter_data_ax[0]), float(filter_data_ax[1]), float(filter_data_ax[2])])
        elif tag == OsimTags.orientation_in_parent:
            data_ax = info.childNodes[0].data.strip().split(" ")
            filter_data_ax = []
            for d in data_ax:
                if len(d) > 0:
                    filter_data_ax.append(d)
            self.orientation_in_parent = np.array([float(filter_data_ax[0]), float(filter_data_ax[1]), float(filter_data_ax[2])])
        elif tag == OsimTags.location:
            data_ax = info.childNodes[0].data.strip().split(" ")
            self.location = np.array([float(data_ax[0]), float(data_ax[1]), float(data_ax[2])])
        elif tag == OsimTags.orientation:
            data_ax = info.childNodes[0].data.strip().split(" ")
            self.orientation = np.array([float(data_ax[0]), float(data_ax[1]), float(data_ax[2])])
        elif tag == OsimTags.CoordinateSet:
            objects = info.childNodes
            for obj in objects:
                if obj.nodeType == Node.ELEMENT_NODE:
                    for ele in obj.childNodes:
                        if ele.nodeType == Node.ELEMENT_NODE:
                            coordinate = OsimCoordinate()
                            coordinate.parse(ele)
                            self.coordinate_set[coordinate.name] = coordinate
        elif tag == OsimTags.reverse:
            data = info.childNodes[0].data.strip()
            self.reverse = OsimTypes.bool(data)
        elif tag == OsimTags.SpatialTransform:
            osim_transform = OsimTransform()
            axis = info.getElementsByTagName(OsimTags.TransformAxis.value)
            for a in axis:
                t = OsimTransformAxis()
                key = None
                if a.hasAttribute(OsimAttributes.name.value):
                    key = OsimTypes.get_type(a.getAttribute(OsimAttributes.name.value))
                data = a.getElementsByTagName(OsimTags.coordinates.value)
                for d in data:
                    if len(d.childNodes) > 0:
                        target = d.childNodes[0].data.strip()
                        t.coordinate = self.coordinate_set[target]

                data = a.getElementsByTagName(OsimTags.axis.value)
                for d in data:
                    data_ax = d.childNodes[0].data.strip().split()
                    t.axis = np.array([float(data_ax[0]), float(data_ax[1]), float(data_ax[2])])

                if "rotation" in key.value:
                    osim_transform.rotation_transforms[key] = t
                elif "translation" in key.value:
                    osim_transform.translation_transforms[key] = t
                else:
                    print(key.value)
            self.transform = osim_transform


class OsimCoordinate(object):
    def __init__(self):
        self.name = ""
        self.motion_type = None
        self.default_value = 0.0
        self.default_speed_value = 0.0
        self.range = {OsimTypes.min: np.NaN, OsimTypes.max: np.NaN}
        self.clamped = True
        self.locked = False
        self.prescribed_function = None
        self.prescribed = False

    def parse(self, info):
        if info.hasAttribute(OsimAttributes.name.value):
            self.name = info.getAttribute(OsimAttributes.name.value)

        data = info.getElementsByTagName(OsimTags.motion_type.value)
        for d in data:
            self.motion_type = OsimTypes.get_type(d.childNodes[0].data.strip())

        data = info.getElementsByTagName(OsimTags.default_value.value)
        for d in data:
            self.default_value = float(d.childNodes[0].data.strip())

        data = info.getElementsByTagName(OsimTags.default_speed_value.value)
        for d in data:
            self.default_speed_value = float(d.childNodes[0].data.strip())

        data = info.getElementsByTagName(OsimTags.range.value)
        for d in data:
            data_ax = d.childNodes[0].data.strip().split(" ")
            self.range[OsimTypes.min] = float(data_ax[0])
            self.range[OsimTypes.max] = float(data_ax[1])

        data = info.getElementsByTagName(OsimTags.clamped.value)
        for d in data:
            self.clamped = OsimTypes.bool(d.childNodes[0].data.strip())

        data = info.getElementsByTagName(OsimTags.locked.value)
        for d in data:
            self.locked = OsimTypes.bool(d.childNodes[0].data.strip())

        data = info.getElementsByTagName(OsimTags.prescribed_function.value)
        for d in data:
            if len(d.childNodes) > 0:
                self.prescribed_function = d.childNodes[0].data.strip()

        data = info.getElementsByTagName(OsimTags.prescribed.value)
        for d in data:
            self.prescribed = OsimTypes.bool(d.childNodes[0].data.strip())


class OsimBody(object):
    def __init__(self):
        self.root_model = None
        self.name = ""
        self.mass = 0.0
        self.mass_center = 0.0
        self.inertia_xx = 0.0
        self.inertia_yy = 0.0
        self.inertia_zz = 0.0
        self.inertia_xy = 0.0
        self.inertia_xz = 0.0
        self.inertia_yz = 0.0
        self.joints = []
        self.visualisation = None
        self.wrap_object = None

    def __str__(self):
        ret = "OsimBody: "+self.name + "\n"
        ret += "> OsimJoint: \n"
        for j in self.joints:
            ret += "|____"+j.__str__()
        return ret

    def parse(self, b):
        if b.hasAttribute(OsimAttributes.name.value):
            self.name = b.getAttribute(OsimAttributes.name.value)

            # Center of mass and inertia components
            mass_element = b.getElementsByTagName(OsimTags.mass.value)
            for m in mass_element:
                self.mass = float(m.childNodes[0].data.strip())

            mass_center_element = b.getElementsByTagName(OsimTags.mass_center.value)
            for m in mass_center_element:
                data_ax = m.childNodes[0].data.strip().split(" ")
                self.mass_center = np.array([float(data_ax[0]), float(data_ax[1]), float(data_ax[2])])

            inertia = b.getElementsByTagName(OsimTags.inertia_xx.value)
            for m in inertia:
                self.inertia_xx = float(m.childNodes[0].data.strip())

            inertia = b.getElementsByTagName(OsimTags.inertia_yy.value)
            for m in inertia:
                self.inertia_yy = float(m.childNodes[0].data.strip())

            inertia = b.getElementsByTagName(OsimTags.inertia_zz.value)
            for m in inertia:
                self.inertia_zz = float(m.childNodes[0].data.strip())

            inertia = b.getElementsByTagName(OsimTags.inertia_xy.value)
            for m in inertia:
                self.inertia_xy = float(m.childNodes[0].data.strip())

            inertia = b.getElementsByTagName(OsimTags.inertia_xz.value)
            for m in inertia:
                self.inertia_xz = float(m.childNodes[0].data.strip())

            inertia = b.getElementsByTagName(OsimTags.inertia_yz.value)
            for m in inertia:
                self.inertia_yz = float(m.childNodes[0].data.strip())

            joint_list = b.getElementsByTagName(OsimTags.joint.value)
            for joints in joint_list:
                jo = joints.childNodes
                for j in jo:
                    if j.nodeType == Node.ELEMENT_NODE:
                        joint = OsimJoint()
                        joint.root_model = self.root_model
                        joint.joint_type = j.nodeName
                        if j.hasAttribute(OsimAttributes.name.value):
                            joint.name = j.getAttribute(OsimAttributes.name.value)
                        joint_info = j.childNodes
                        last = None
                        for t in joint_info:
                            if t.nodeType == Node.ELEMENT_NODE:
                                target = t.nodeName.lower()
                                if "transform" not in target and OsimTags.is_tag(t.nodeName):
                                    joint.parse(t, OsimTags.get_tag(t.nodeName))
                                else:
                                    last = t
                        if last is not None:
                            joint.parse(last, OsimTags.get_tag(last.nodeName))


                        self.joints.append(joint)


class OsimModel(object):
    def __init__(self):
        self.name = ""
        self.credits = ""
        self.publications = ""
        self.length_units = ""
        self.force_units = ""
        self.rotation_unit = OsimTypes.radian
        self.gravity = None
        self.body_set = {}
        self.constraint_set = {}
        self.force_set = {}
        self.marker_set = {}
        self.contact_geometry_set = []
        self.controller_set = []
        self.probeSet = []
        self.component_set = []

    def find_body(self, body_name):
        return self.body_set[body_name]

    def build_simulaThor_Model(self, model):
        self.create_bodies(model)
        self.create_joints(model)
        return model

    def create_bodies(self, model):
        for b in self.body_set:
            model.body(self.body_set[b])
        pass

    def create_joints(self, model):
        for b in self.body_set:
            model.link_bodies(self.body_set[b].joints)

        model.init_joints()
        pass


class OsimLezer(xml.sax.ContentHandler):
    def __init__(self, filename=None):
        self.CurrentData = ""
        self.OpenSimDocument_version = ""
        self.model = OsimModel()
        self.osim_doc = None
        if filename is not None:
            self.parse(filename)

    def parse(self, filename):
        if filename is None:
            return
        DOMTree = xml.dom.minidom.parse(filename)
        self.osim_doc = DOMTree.documentElement
        if self.osim_doc.hasAttribute(OsimAttributes.version.value):
            self.OpenSimDocument_version = self.osim_doc.getAttribute(OsimAttributes.version.value)
            print("|---> Opensim Doc Version: ", self.OpenSimDocument_version)

        # read model name
        model_element = self.osim_doc.getElementsByTagName(OsimTags.Model.value)
        self.model.name = model_element[0].getAttribute(OsimAttributes.name.value)

        # read credits
        credit_element = self.osim_doc.getElementsByTagName(OsimTags.credits.value)
        for c in credit_element:
            self.model.credits = c.childNodes[0].data

        # read publications
        pub_element = self.osim_doc.getElementsByTagName(OsimTags.publications.value)
        for c in pub_element:
            self.model.publications = c.childNodes[0].data

        len_element = self.osim_doc.getElementsByTagName(OsimTags.length_units.value)
        for c in len_element:
            self.model.length_units = c.childNodes[0].data

        force_unit_element = self.osim_doc.getElementsByTagName(OsimTags.force_units.value)
        for c in force_unit_element:
            self.model.force_units = c.childNodes[0].data

        gravity_element = self.osim_doc.getElementsByTagName(OsimTags.gravity.value)
        for c in gravity_element:
            data = c.childNodes[0].data
            data_ax = data.strip().split(" ")
            vec = np.array([float(data_ax[0]), float(data_ax[1]), float(data_ax[2])])
            self.model.gravity = vec

        self.process_body_set(self.osim_doc.getElementsByTagName(OsimTags.body_set.value))
        self.process_marker_set(self.osim_doc.getElementsByTagName(OsimTags.marker_set.value))
        print("|---> Model Name: ", self.model.name)
        return self.model

    def process_body_set(self, body_set_elements):
        for c in body_set_elements:
            obj = c.getElementsByTagName(OsimTags.objects.value)
            for d in obj:
                bodies = d.getElementsByTagName(OsimTags.body.value)
                for b in bodies:
                    body = OsimBody()
                    body.root_model = self.model
                    body.parse(b)
                    self.model.body_set[body.name] = body

    def process_marker_set(self, marker_elements):
        #print(marker_elements)
        for c in marker_elements:
            obj = c.getElementsByTagName(OsimTags.objects.value.lower())
            for d in obj:
                markers = d.getElementsByTagName(OsimTags.Marker.value)
                #print(markers)
                for m in markers:
                    marker = OsimMarker()
                    marker.root_model = self.model
                    marker.parse(m)


class STOBasicReader(object):
    @staticmethod
    def read_file(ik_data):
        ik = []
        with open(ik_data, newline='') as csvfile:
            ik_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            i = 0
            for a in ik_reader:
                if len(a) > 3:
                    ik.append(a)
                i += 1

        labels = ik[0]
        data = ik[1:-1]
        ik_base = {}
        for i in data:
            for j in range(0, len(labels)):
                if ik_base.get(labels[j]) is None:
                    ik_base[labels[j]] = [float(i[j].strip())]
                else:
                    ik_base[labels[j]].append(float(i[j].strip()))
        return ik_base

