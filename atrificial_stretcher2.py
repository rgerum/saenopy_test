import saenopy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from saenopy.materials import SemiAffineFiberMaterial
import os
from saenopy.getDeformations import interpolate_different_mesh
from saenopy.multigridHelper import getScaledMesh, createMesh
import time

import numpy as np

x = np.load("C12-new-mean-stretch.npy")
y = np.load("C12-new-mean-vertical.npy")


def getNearestNode(R, point):
    return np.argmin(np.linalg.norm(R - np.array(point)[None, :], axis=1))


class Strecher:
    stretch_values = {}

    def __init__(self, Wx, Wy, Wz, element_width=500e-6):
        M = saenopy.Solver()
        self.Wx = Wx
        self.Wy = Wy
        self.Wz = Wz
        R, T = createMesh(element_width=element_width, box_width=(Wx, Wy, Wz))
        R[:, 0] -= np.min(R[:, 0])
        R[:, 0] -= np.max(R[:, 0])/2
        R[:, 1] -= np.min(R[:, 1])
        R[:, 1] -= np.max(R[:, 1])/2
        R[:, 2] -= np.min(R[:, 2])
        M.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00215, 0.032, 0.055))
        M.setNodes(R)
        M.setTetrahedra(T)

        stretch = 0

        bcond_disp = np.zeros_like(R) * np.nan
        bcond_force = np.zeros_like(R)
        minR = np.min(R, axis=0)
        maxR = np.max(R, axis=0)
        width = 0.5e-6
        wall_x0 = (R[:, 0] < minR[0] + width)
        wall_x1 = (R[:, 0] > maxR[0] - width)
        wall_y0 = (R[:, 1] < minR[1] + width)
        wall_y1 = (R[:, 1] > maxR[1] - width)
        wall_z0 = (R[:, 2] < minR[2] + width)

        bcond_force[wall_x0 | wall_x1] = np.nan
        bcond_force[wall_y0 | wall_y1] = np.nan
        bcond_force[wall_z0] = np.nan
        bcond_disp[wall_x0] = np.array([-stretch / 2, 0, 0])
        bcond_disp[wall_x1] = np.array([stretch / 2, 0, 0])
        bcond_disp[wall_y0] = R[wall_y0, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
        bcond_disp[wall_y1] = R[wall_y1, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])
        bcond_disp[wall_z0] = R[wall_z0, 0:1] / maxR[1] * np.array([[stretch / 2, 0, 0]])

        U = M.U[:]
        U[:, 0] = R[:, 0] / maxR[1] * stretch / 2
        M.setBoundaryCondition(bcond_disp, bcond_force)
        M.setInitialDisplacements(U)

        self.M = M
        self.M._check_relax_ready()

        self.M._prepare_temporary_quantities()
        self.stretch_values = {}

    def setMaterialModel(self, material):
        self.M.setMaterialModel(material)

    def getStretch(self, stretch, i_max=300):
        stretch = stretch - 1
        stretch_m = stretch * Wx
        print(stretch, stretch_m)
        R = self.M.R
        minR = np.min(R, axis=0)
        maxR = np.max(R, axis=0)
        if stretch not in self.stretch_values:
            bcond_disp = np.zeros_like(R) * np.nan
            bcond_force = np.zeros_like(R)

            width = 0.5e-6
            wall_x0 = (R[:, 0] < minR[0] + width)
            wall_x1 = (R[:, 0] > maxR[0] - width)
            wall_y0 = (R[:, 1] < minR[1] + width)
            wall_y1 = (R[:, 1] > maxR[1] - width)
            wall_z0 = (R[:, 2] < minR[2] + width)

            bcond_force[wall_x0 | wall_x1] = np.nan
            bcond_force[wall_y0 | wall_y1] = np.nan
            bcond_force[wall_z0] = np.nan
            bcond_disp[wall_x0] = np.array([-stretch_m / 2, 0, 0])
            bcond_disp[wall_x1] = np.array([stretch_m / 2, 0, 0])
            bcond_disp[wall_y0] = R[wall_y0, 0:1] / maxR[1] * np.array([[stretch_m / 2, 0, 0]])
            bcond_disp[wall_y1] = R[wall_y1, 0:1] / maxR[1] * np.array([[stretch_m / 2, 0, 0]])
            bcond_disp[wall_z0] = R[wall_z0, 0:1] / maxR[1] * np.array([[stretch_m / 2, 0, 0]])

            self.bcond_force = bcond_force

            U = self.M.U.copy()
            U[:, 0] = R[:, 0] / maxR[1] * stretch_m / 2
            self.stretch_values[stretch] = [bcond_disp, U]
        bcond_disp, U = self.stretch_values[stretch]

        self.M.setBoundaryCondition(bcond_disp, self.bcond_force)
        self.M.setInitialDisplacements(U)
        import time
        t = time.time()
        relrec = self.M.solve_boundarycondition(i_max=i_max, callback=callback, rel_conv_crit=-np.inf)
        print(time.time()-t, relrec[-1])
        return self.M.U[getNearestNode(R, [0, 0, maxR[2]]), 2] / maxR[2] + 1

    def getStretchArray(self, stretch, i_max=300):
        return np.array([self.getStretch(s, i_max) for s in stretch])


Wx = 2e-2*1.05
Wy = 2e-2
V = 700e-6*1e-3
Wz = V/(Wx*Wy)
S = Strecher(Wx=Wx, Wy=Wy, Wz=Wz)


R = S.M.R
minR = np.min(R, axis=0)
maxR = np.max(R, axis=0)
bcond_disp = np.zeros_like(R) * np.nan
bcond_force = np.zeros_like(R)

width = 0.5e-6
wall_x0 = (R[:, 0] < minR[0] + width)
wall_x1 = (R[:, 0] > maxR[0] - width)
wall_y0 = (R[:, 1] < minR[1] + width)
wall_y1 = (R[:, 1] > maxR[1] - width)
wall_z0 = (R[:, 2] < minR[2] + width)

bcond_force[wall_x0 | wall_x1] = np.nan
bcond_force[wall_y0 | wall_y1] = np.nan
bcond_force[wall_z0] = np.nan
it = 20
start_time = time.time()
def callback(M, relrec):
    global start_time
    index = len(relrec) // it
    modu = len(relrec) % it
    if modu == 0:
        d.append(M.U[getNearestNode(R, [0, 0, maxR[2]]), 2] / maxR[2] + 1)
        print(index, len(relrec), it, xx[index] - 1, d[-1], relrec[-1], time.time() - start_time)
        start_time = time.time()
        stretch = xx[index] - 1
        stretch_m = stretch * Wx
        bcond_disp = M.U_fixed
        bcond_disp[wall_x0] = np.array([-stretch_m / 2, 0, 0])
        bcond_disp[wall_x1] = np.array([stretch_m / 2, 0, 0])
        bcond_disp[wall_y0] = R[wall_y0, 0:1] / maxR[1] * np.array([[stretch_m / 2, 0, 0]])
        bcond_disp[wall_y1] = R[wall_y1, 0:1] / maxR[1] * np.array([[stretch_m / 2, 0, 0]])
        bcond_disp[wall_z0] = R[wall_z0, 0:1] / maxR[1] * np.array([[stretch_m / 2, 0, 0]])
        M.U[~M.var] = bcond_disp[~M.var]

S.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00055, 0.032, 0.055))
#d = S.getStretchArray([1.05, 1.1, 1.15])
xx = np.arange(1, 1.20, 0.002)
d = []
try:
    S.getStretch(1, i_max=len(xx)*it)
except:
    np.save("d_1_1.20_0.002_imax20_k1449_0.00055_0.0032_0.055.npy", d)
#d = S.getStretchArray(xx, i_max=2)
#d2 = S.getStretchArray(xx, i_max=1)
plt.plot(xx[:len(d)], d)
plt.plot(xx[:len(d2)], d2)
plt.plot(xx[:len(d3)], d3)
plt.plot(x, y)
#d2 = d.copy()

d = np.load("d_1_1.20_0.002_imax20_k1449_0.00055_0.0032_0.055.npy")
plt.plot(xx[:len(d)], d, label=f"lambda_s {0.0032}")
d = np.load("d_1_1.20_0.002_imax20_k1449_0.00055_0.032_0.055.npy")
plt.plot(xx[:len(d)], d, label=f"lambda_s {0.032}")
d = np.load("d_1_1.20_0.002_imax20_k1449_0.00055_0.32_0.055.npy")
plt.plot(xx[:len(d)], d, label=f"lambda_s {0.32}")
d = np.load("d_1_1.20_0.002_imax20_k1449_0.00055_320_0.055.npy")
plt.plot(xx[:len(d)], d, label=f"lambda_s {320}")
plt.legend()

d = np.load("d_1_1.20_0.002_imax20_k549_0.00055_0.032_0.055.npy")
d2 = np.load("d_1_1.20_0.002_imax20_k1449_0.00055_0.032_0.055.npy")
d3 = np.load("d_1_1.08_0.002_imax20_k2049_0.00055_0.032_0.055.npy")
#np.save("d_1_1.20_0.002_imax20_k1449_0.00055_0.32_0.055.npy", d)
#d2 = np.load("d2.npy")
#d3 = np.load("d_1_1.08_0.002_imax2.npy")

from saenopy import macro
lambda_h = np.arange(1-0.05, 1+0.17, 0.01)
lambda_v = np.arange(0, 1.1, 0.001)

x2, y2 = macro.getStretchThinning(lambda_h, lambda_v, S.M.material_model)
plt.plot(x2, y2, lw=3, label="model")

x2, y2 = macro.getStretchThinning(lambda_h, lambda_v, SemiAffineFiberMaterial(449, 0.00055, 0.032, 0.055))
plt.plot(x2, y2, lw=3, label="model")
x2, y2 = macro.getStretchThinning(lambda_h, lambda_v, SemiAffineFiberMaterial(1449, 0.00055, 0.32, 0.055))
plt.plot(x2, y2, lw=3, label="model")
x2, y2 = macro.getStretchThinning(lambda_h, lambda_v, SemiAffineFiberMaterial(2449, 0.00055, 3.2, 0.055))
plt.plot(x2, y2, lw=3, label="model")

S.setMaterialModel(SemiAffineFiberMaterial(1449, 0.00055, 0.032, 0.055))
plt.show()
