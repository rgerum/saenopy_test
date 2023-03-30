import os
import numpy as np
import typing
from typing import List
try:
    from typing import _GenericAlias as _GenericAlias
except ImportError:
    from typing import GenericMeta as _GenericAlias

from .multigridHelper import makeBoxmeshCoords, makeBoxmeshTets, setActiveFields


class Saveable:
    __save_parameters__ = []

    def __init__(self, **kwargs):
        for name in kwargs:
            if name in self.__save_parameters__:
                setattr(self, name, kwargs[name])

    def to_dict(self):
        data = {}
        for param in self.__save_parameters__:
            attribute = getattr(self, param, None)
            if attribute is not None:
                if getattr(attribute, "to_dict", None) is not None:
                    data[param] = getattr(attribute, "to_dict")()
                elif isinstance(attribute, list) and len(attribute) and (getattr(attribute[0], "to_dict", None) is not None or attribute[0] is None):
                    my_list = []
                    for attr in attribute:
                        value = attr
                        if getattr(attribute[0], "to_dict", None):
                            value = getattr(attr, "to_dict")()
                        if attr is None:
                            value = "__NONE__"
                        my_list.append(value)
                    data[param] = my_list
                elif attribute is None:
                    data[param] = "__NONE__"
                else:
                    data[param] = attribute
        return data

    def save(self, filename: str):
        data = self.to_dict()

        if filename.endswith("h5py") or filename.endswith("h5"):
            return dict_to_h5(filename, flatten_dict(data))

        #np.savez(filename, **data)
        np.lib.npyio._savez(filename, [], flatten_dict(data), True, allow_pickle=False)

    @classmethod
    def from_dict(cls, data_dict):
        types = typing.get_type_hints(cls)
        data = {}
        for name in data_dict:
            if isinstance(data_dict[name], np.ndarray) and len(data_dict[name].shape) == 0:
                data[name] = data_dict[name][()]
            else:
                data[name] = data_dict[name]
            if name in types:
                if getattr(types[name], "from_dict", None) is not None:
                    data[name] = types[name].from_dict(data[name])
                elif typing.get_origin(types[name]) is list:
                    if isinstance(data[name], dict):
                        data[name] = typing.get_args(types[name])[0].from_dict(data[name])
                    else:
                        data[name] = [None if d == "__NONE__" else typing.get_args(types[name])[0].from_dict(d) for d in data[name]]
        return cls(**data)

    @classmethod
    def load(cls, filename):
        if str(filename).endswith(".h5py") or str(filename).endswith(".h5"):
            import h5py
            data = h5py.File(filename, "a")
            result = cls.from_dict(unflatten_dict_h5(data))
        else:
            data = np.load(filename, allow_pickle=False)

            result = cls.from_dict(unflatten_dict(data))
        if getattr(result, 'on_load', None) is not None:
            getattr(result, 'on_load')(filename)
        return result

def flatten_dict(data):
    result = {}

    def print_content(data, prefix):
        if isinstance(data, list):  # and not isinstance(data[0], (int, float)):
            result[prefix] = "list"
            for name, d in enumerate(data):
                if d is None:
                    d = "__NONE__"
                print_content(d, f"{prefix}/{name}")
            return
        if isinstance(data, tuple):  # and not isinstance(data[0], (int, float)):
            result[prefix] = "tuple"
            for name, d in enumerate(data):
                if d is None:
                    d = "__NONE__"
                print_content(d, f"{prefix}/{name}")
            return
        if isinstance(data, dict):  # and not isinstance(data[0], (int, float)):
            result[prefix] = "dict"
            for name, d in data.items():
                if d is None:
                    d = "__NONE__"
                print_content(d, f"{prefix}/{name}")
            return
        result[prefix] = data

    for name, d in data.items():
        print_content(d, name)

    return result

def unflatten_dict(data):
    result = {}
    for name, item in data.items():
        if item.shape == ():
            item = item[()]
        if isinstance(item, str) and item == "list":
            item = []
        if isinstance(item, str) and item == "dict":
            item = {}
        if isinstance(item, str) and item == "tuple":
            item = ()

        names = name.split("/")

        hierarchy = [result]
        r = result
        for name in names[:-1]:
            try:
                r = r[name]
            except TypeError:
                r = r[int(name)]
            except KeyError:
                continue
            hierarchy.append(r)

        if isinstance(r, list):
            r += [item]
        elif isinstance(r, tuple):
            try:
                hierarchy[-2][names[-2]] = r + (item,)
            except TypeError:
                hierarchy[-2][int(names[-2])] = r + (item,)
        else:
            if isinstance(item, str) and item == "__NONE__":
                item = None
            r[names[-1]] = item

    return result


def dict_to_h5(filename, data):
    import h5py
    with h5py.File(filename, "w") as f:
        for key in data.keys():
            if isinstance(data[key], str):
                if str(data[key]) == "list" or str(data[key]) == "dict" or str(data[key]) == "tuple":
                    grp = f.create_group(key)
                    grp.attrs["type"] = str(data[key])
                    continue
                dset = f.create_dataset(key, data=str(data[key]))
            elif str(data[key].dtype).startswith("<U"):
                dset = f.create_dataset(key, data=str(data[key]))
            else:
                try:
                    f.create_dataset(key, data=data[key], compression="gzip")
                except TypeError:
                    f.create_dataset(key, data=data[key])


def unflatten_dict_h5(data):
    import h5py
    if isinstance(data, h5py.Group):
        if data.attrs.get("type") == "list":
            result = [unflatten_dict_h5(v) for v in data.values()]
        elif data.attrs.get("type") == "tuple":
            result = tuple([unflatten_dict_h5(v) for v in data.values()])
        else:
            result = {k: unflatten_dict_h5(v) for k, v in data.items()}
        return result
    else:
        if data.shape == ():
            try:
                return data.asstr()[()]
            except TypeError:
                return data[()]
        else:
            return data

def load(filename, *args, **kwargs):
    file2 = filename[:-4] + ".npy"
    if not os.path.exists(file2) or os.path.getmtime(filename) > os.path.getmtime(file2):
        print("Load changed file", filename)
        data = np.loadtxt(filename, *args, **kwargs)
        np.save(file2, data)
    else:
        data = np.load(file2)
    return data


def makeBoxmesh(mesh, CFG):
    mesh.currentgrain = 1

    nx = CFG["BM_N"]
    dx = CFG["BM_GRAIN"]

    rin = CFG["BM_RIN"]
    mulout = CFG["BM_MULOUT"]
    rout = nx * dx * 0.5

    if rout < rin:
        print("WARNING in makeBoxmesh: Mesh BM_RIN should be smaller than BM_MULOUT*BM_GRAIN*0.5")

    print("coords")
    mesh.setNodes(makeBoxmeshCoords(dx, nx, rin, mulout))
    print("tets")
    mesh.setTetrahedra(makeBoxmeshTets(nx, mesh.currentgrain))
    print("var")
    mesh.var = setActiveFields(nx, mesh.currentgrain, True)
    print("done")


def loadMeshCoords(fcoordsname):
    """
    Load the nodes. Each line represents a node and has 3 float entries for the x, y, and z coordinates of the
    node.
    """

    # load the node file
    data = load(fcoordsname, dtype=float)

    # check the data
    assert data.shape[1] == 3, "coordinates in " + fcoordsname + " need to have 3 columns for the XYZ"
    print("%s read (%d entries)" % (fcoordsname, data.shape[0]))

    return data


def loadMeshTets(ftetsname):
    """
    Load the tetrahedrons. Each line represents a tetrahedron. Each line has 4 integer values representing the node
    indices.
    """
    # load the data
    data = load(ftetsname, dtype=int)

    # check the data
    assert data.shape[1] == 4, "node indices in " + ftetsname + " need to have 4 columns, the indices of the nodes of the 4 corners fo the tetrahedron"
    print("%s read (%d entries)" % (ftetsname, data.shape[0]))

    # the loaded data are the node indices but they start with 1 instead of 0 therefore "-1"
    return data - 1


def loadBeams(self, fbeamsname):
    return np.loadtxt(fbeamsname)


def loadBoundaryConditions(dbcondsname, N_c=None):
    """
    Loads a boundary condition file "bcond.dat".

    It has 4 values in each line.
    If the last value is 1, the other 3 define a force on a variable node
    If the last value is 0, the other 3 define a displacement on a fixed node
    """
    # load the data in the file
    data = np.loadtxt(dbcondsname)
    assert data.shape[1] == 4, "the boundary conditions need 4 columns"
    if N_c is not None:
        assert data.shape[0] == N_c, "the boundary conditions need to have the same count as the number of nodes"
    print("%s read (%d x %d entries)" % (dbcondsname, data.shape[0], data.shape[1]))

    # the last column is a bool whether the node is fixed or not
    var = data[:, 3] > 0.5
    # if it is fixed, the other coordinates define the displacement
    U = np.zeros((var.shape[0], 3))*np.nan
    U[~var] = data[~var, :3]
    # if it is variable, the given vector is the force on the node
    f_ext = np.zeros((var.shape[0], 3))*np.nan
    f_ext[var] = data[var, :3]

    # update the connections (as they only contain non-fixed nodes)
    return var, U, f_ext


def loadConfiguration(Uname, N_c=None):
    """
    Load the displacements for the nodes. The file has to have 3 columns for the displacement in XYZ and one
    line for each node.
    """
    data = np.loadtxt(Uname)
    assert data.shape[1] == 3, "the displacement file needs to have 3 columnds"
    if N_c is not None:
        assert data.shape[0] == N_c, "there needs to be a displacement for each node"
    print("%s read (%d entries)" % (Uname, data.shape[0]))

    # store the displacement
    return data