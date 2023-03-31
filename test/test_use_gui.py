#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from mock_dir import MockDir
import tifffile
from saenopy.gui_master import MainWindow
from qtpy import QtWidgets
import sys
np.random.seed(1234)

import sys
from qtpy import QtCore, QtWidgets, QtGui
from pathlib import Path
import multiprocessing

from skimage.filters import ridges, thresholding
from saenopy.gui import QtShortCuts
from saenopy.gui_deformation_whole2 import MainWindow as SolverMain
from saenopy.gui_deformation_spheriod import MainWindow as SpheriodMain
from saenopy.gui_orientation import MainWindow as OrientationMain
from saenopy.gui.resources import resource_path, resource_icon


def create_tif(filename, y=20, x=10, z=1, rgb=None):
    with tifffile.TiffWriter(filename) as tif:
        for i in range(z):
            if rgb is None:
                tif.write(np.random.rand(y, x))
            else:
                tif.write(np.random.rand(y, x, rgb))


def sf4(x):
    if isinstance(x, float):
        x = float(np.format_float_positional(x, precision=4, unique=False, fractional=False,trim='k'))
        return x
    return [sf4(xx) for xx in x]


def test_stack():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    import pyvista as pv
    p = pv.Plotter(off_screen=True)
    p.add_mesh(pv.Cylinder(), color="tan", show_edges=True)
    p.show(screenshot="test.png")

    from pyvistaqt import QtInteractor
    QtInteractor(window, auto_update=False)
    window.changedTab(1)
    #return
    file_structure = {
        "tmp": {
            "run-1": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
            "run-1-reference": [f"Pos004_S001_z{z:03d}_ch00.tif" for z in range(50)],
        }
    }
    with MockDir(file_structure, lambda file: create_tif(file, x=50, y=50)):
        #app = QtWidgets.QApplication(sys.argv)
        #window = MainWindow()  # gui_master.py:MainWindow
        solver = window.solver  # gui_deformation_whole2.py:MainWindow
        batch_evaluate = solver.deformations  # gui_solver/BatchEvaluate.py:BatchEvaluate

        # get the input
        from saenopy import get_stacks
        results = get_stacks("tmp/run-1/Pos004_S001_z{z}_ch00.tif", "tmp/run-1", [1, 1, 1],
                             reference_stack="tmp/run-1-reference/Pos004_S001_z{z}_ch00.tif")
        # bundle the images
        results[0].stack[0].pack_files()
        results[0].stack_reference.pack_files()
        # add them to the gui and select them
        batch_evaluate.list.addData(results[0].output, True, results[0], "#FF0000")
        batch_evaluate.list.setCurrentRow(0)

        # change some parameters
        batch_evaluate.sub_module_deformation.setParameter("win_um", 30)
        batch_evaluate.sub_module_regularize.setParameter("i_max", 10)
        print("params A", getattr(batch_evaluate.sub_module_deformation.result, batch_evaluate.sub_module_deformation.params_name + "_tmp"))
        print("params C", getattr(batch_evaluate.sub_module_regularize.result, batch_evaluate.sub_module_regularize.params_name + "_tmp"))

        # schedule to run all
        batch_evaluate.run_all()

        # wait until they are processed
        while batch_evaluate.current_task_id < len(batch_evaluate.tasks):
            app.processEvents()

        # check the result
        M = results[0].solver[0]
        print(M.U[M.reg_mask])
        print(results[0].solver[0].U.shape)
        assert sf4(M.U[M.reg_mask][0]) == sf4([-2.01259036e-38, -1.96865342e-38, -4.92921492e-38])
