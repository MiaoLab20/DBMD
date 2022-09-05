from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
fix_gpu()

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
mpl.use("Agg")

import pandas as pd
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, Reduction
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp
from deeplearningmodel2 import *
from math import sqrt
from DLGaMDEquilibrationIntegrator import *
from utils import *

tfpd = tfp.distributions
tfpl = tfp.layers

from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
"""
OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
"""
prmtop = AmberPrmtopFile("gcaa_unfold.top")
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=8*angstrom, constraints=HBonds)
# system = prmtop.createSystem(implicitSolvent=GBn2, implicitSolventKappa=1.0/nanometer, soluteDielectric=1.0, solventDielectric=78.5)
set_dihedral_group(system)

integrator = DualDeepLearningGaMDEquilibration(0.002*picoseconds, 300*kelvins)

isExist = os.path.exists("equil.mdout")
if isExist:
    os.remove("equil.mdout")
simulation = Simulation(prmtop.topology, system, integrator)
simulation.loadState("cmd.rst")
simulation.reporters.append(DCDReporter("equil.dcd", 400))
simulation.reporters.append(ExpandedStateDataReporter(system, "equil.mdout", 400, step=True, temperature=True,
                                                    brokenOutForceEnergies=True, potentialEnergy=True,
                                                    totalEnergy=True, density=True, separator="\t",
                                                    speed=True, remainingTime=True, totalSteps=25000000))
isExist = os.path.exists("equil.log")
if isExist:
    os.remove("equil.log")
gamdLog = open("equil.log", "w")
gamdLog.write("# Gaussian accelerated Molecular Dynamics log file\n")
gamdLog.write("# All energy terms are stored in unit of kcal/mol\n")
gamdLog.write("# ntwx,total_nstep,Total-Energy,Dihedral-Energy,Total-Force-Weight,Dihedral-Force-Weight,"
              "Total-Boost,Dihedral-Boost,Total-Harmonic-Force-Constant,Dihedral-Harmonic-Force-Constant,"
              "Minimum-Total-Energy,Maximum-Total-Energy,Minimum-Dihedral-Energy,Maximum-Dihedral-Energy"
              "Total-Boost-Mode,Dihedral-Boost-Mode\n")
gamdLog.close()

DihedralEnergies, TotalEnergies = [], []
for step in np.arange(0, 62500):
    simulation.step(400)
    if (step + 1) % 400 == 0: 
        simulation.saveState("equil.rst")
    gamdLog = open("equil.log", "a")
    gamdLog.write(str(400) + "\t" + str((step+1)*400) + "\t"
                + str(integrator.getGlobalVariableByName("TotalEnergy") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("DihedralEnergy") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("TotalForceScalingFactor")) + "\t"
                + str(integrator.getGlobalVariableByName("DihedralForceScalingFactor")) + "\t"
                + str(integrator.getGlobalVariableByName("TotalBoostPotential") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("DihedralBoostPotential") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("Totalk0")) + "\t"
                + str(integrator.getGlobalVariableByName("Dihedralk0")) + "\t"
                + str(integrator.getGlobalVariableByName("VminP") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("VmaxP") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("VminD") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("VmaxD") / 4.184) + "\t"
                + str(integrator.getGlobalVariableByName("TotalBoostMode")) + "\t"
                + str(integrator.getGlobalVariableByName("DihedralBoostMode")) + "\n")
    gamdLog.close()

    dihedralboost = np.absolute(dihadjustboost + modeldih.predict(np.asarray([integrator.getGlobalVariableByName("DihedralEnergy") / 4.184])).flatten()[0] * 4.184)
    integrator.setGlobalVariableByName("DihedralBoostPotential", dihedralboost)
    totalboost = np.absolute(totadjustboost + modeltot.predict(np.asarray([integrator.getGlobalVariableByName("TotalEnergy") / 4.184])).flatten()[0] * 4.184)
    integrator.setGlobalVariableByName("TotalBoostPotential", totalboost)

    DihedralEnergies.append(integrator.getGlobalVariableByName("DihedralEnergy") / 4.184)
    TotalEnergies.append(integrator.getGlobalVariableByName("TotalEnergy") / 4.184)
    if (step + 1) % 250 == 0:
        min_dihedral, max_dihedral = min(DihedralEnergies) * 4.184, max(DihedralEnergies) * 4.184
        min_total, max_total = min(TotalEnergies) * 4.184, max(TotalEnergies) * 4.184
        integrator.setGlobalVariableByName("VminD", min_dihedral)
        integrator.setGlobalVariableByName("VmaxD", max_dihedral)
        integrator.setGlobalVariableByName("VminP", min_total)
        integrator.setGlobalVariableByName("VmaxP", max_total)
        DihedralEnergies, TotalEnergies = [], []
