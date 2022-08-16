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

column_names = ["Parameters", "Values"]
cmdRestart = pd.read_csv("cmd-restart.dat", names=column_names, header=0, delimiter="\t")
boost_parameters = cmdRestart.Values.to_list()
boost_parameters = np.array(boost_parameters)

nbsteps = boost_parameters[0]
dihedralEnergy, totalEnergy = boost_parameters[1] * 4.184, boost_parameters[6] * 4.184
min_dihedral, max_dihedral = boost_parameters[2] * 4.184, boost_parameters[3] * 4.184
min_total, max_total = boost_parameters[7] * 4.184, boost_parameters[8] * 4.184

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, Reduction
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp
from deeplearningmodel import *
from math import sqrt
from utils import *

tfpd = tfp.distributions
tfpl = tfp.layers

adjust_boost = 0.5 * 4.184
totalcp_path = "cmd-cptot/cptot-{epoch:04d}.ckpt"
totalcp_dir = os.path.dirname(totalcp_path)
latest_totalcp = tf.train.latest_checkpoint(totalcp_dir)
modeltot = fullModel(nbsteps)
modeltot.load_weights(latest_totalcp)
totalboost = np.absolute(adjust_boost + modeltot.predict(np.asarray([totalEnergy])).flatten()[0] * 4.184)

dihedralcp_path = "cmd-cpdih/cpdih-{epoch:04d}.ckpt"
dihedralcp_dir = os.path.dirname(dihedralcp_path)
latest_dihedralcp = tf.train.latest_checkpoint(dihedralcp_dir)
modeldih = fullModel(nbsteps)
modeldih.load_weights(latest_dihedralcp)
dihedralboost = np.absolute(adjust_boost + modeldih.predict(np.asarray([dihedralEnergy])).flatten()[0] * 4.184)

from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from utils import *
import sys, os
"""
OpenMM Custom Integrator Manual: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
"""
class DualDeepLearningGaMDEquilibration(CustomIntegrator):
    def __init__(self, dt=0.002*picoseconds, temperature=300*kelvins):
        CustomIntegrator.__init__(self, dt)

        self.collision_rate = 1/picosecond
        self.kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        self.temperature = temperature
        self.thermal_energy = self.kB * self.temperature

        self.addGlobalVariable("thermal_energy", self.thermal_energy)
        self.addGlobalVariable("collision_rate", self.collision_rate)
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        self.addGlobalVariable("DihedralEnergy", 0.0)
        self.addGlobalVariable("VminD", min_dihedral)
        self.addGlobalVariable("VmaxD", max_dihedral)
        self.addGlobalVariable("DihedralBoostPotential", dihedralboost)
        self.addGlobalVariable("Dihedralk0prime", 0.0)
        self.addGlobalVariable("Dihedralk0doubleprime", 0.0)
        self.addGlobalVariable("Dihedralk0", 0.0)
        self.addGlobalVariable("DihedralRefEnergy", 0.0)
        self.addGlobalVariable("DihedralForceScalingFactor", 1.0)
        self.addGlobalVariable("DihedralBoostMode", 0.0)
        self.addGlobalVariable("TotalEnergy", 0.0)
        self.addGlobalVariable("VminP", min_total)
        self.addGlobalVariable("VmaxP", max_total)
        self.addGlobalVariable("TotalBoostPotential", totalboost)
        self.addGlobalVariable("Totalk0prime", 0.0)
        self.addGlobalVariable("Totalk0doubleprime", 0.0)
        self.addGlobalVariable("Totalk0", 0.0)
        self.addGlobalVariable("TotalRefEnergy", 0.0)
        self.addGlobalVariable("TotalForceScalingFactor", 1.0)
        self.addGlobalVariable("TotalBoostMode", 0.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()

        self.addComputeGlobal("DihedralEnergy", "energy2")
        Dihedralk0_expr_prime = "((sqrt(2*DihedralBoostPotential*(VmaxD-VminD))-sqrt(2*DihedralBoostPotential*(VmaxD-VminD)-4*(VminD-DihedralEnergy)*(VmaxD-VminD)))/(2*(VminD-DihedralEnergy)))^2"
        Dihedralk0_expr_doubleprime = "((sqrt(2*DihedralBoostPotential*(VmaxD-VminD))+sqrt(2*DihedralBoostPotential*(VmaxD-VminD)-4*(VminD-DihedralEnergy)*(VmaxD-VminD)))/(2*(VminD-DihedralEnergy)))^2"
        self.addComputeGlobal("Dihedralk0prime", Dihedralk0_expr_prime)
        self.addComputeGlobal("Dihedralk0doubleprime", Dihedralk0_expr_doubleprime)
        self.addComputeGlobal("Dihedralk0", "min(Dihedralk0prime, Dihedralk0doubleprime)")
        self.addComputeGlobal("DihedralRefEnergy", "VminD+(VmaxD-VminD)/Dihedralk0")
        self.addComputeGlobal("DihedralBoostMode", "2.0")
        self.beginIfBlock("Dihedralk0 >= 1.0")
        self.addComputeGlobal("DihedralRefEnergy", "VmaxD")
        self.addComputeGlobal("Dihedralk0", "2*DihedralBoostPotential*(VmaxD-VminD)/(DihedralRefEnergy-DihedralEnergy)^2")
        self.addComputeGlobal("DihedralBoostMode", "1.0")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy < DihedralRefEnergy")
        self.addComputeGlobal("DihedralForceScalingFactor", "1-Dihedralk0*(DihedralRefEnergy-DihedralEnergy)/(VmaxD-VminD)")
        self.addComputeGlobal("DihedralEnergy", "DihedralEnergy+DihedralBoostPotential")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy >= DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.0")
        self.addComputeGlobal("Dihedralk0", "0.0")
        self.addComputeGlobal("DihedralForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("VminD", "min(DihedralEnergy, VminD)")
        self.addComputeGlobal("VmaxD", "max(DihedralEnergy, VmaxD)")

        self.addComputeGlobal("TotalEnergy", "energy0+DihedralBoostPotential")
        Totalk0_expr_prime = "((sqrt(2*TotalBoostPotential*(VmaxP-VminP))-sqrt(2*TotalBoostPotential*(VmaxP-VminP)-4*(VminP-TotalEnergy)*(VmaxP-VminP)))/(2*(VminP-TotalEnergy)))^2"
        Totalk0_expr_doubleprime = "((sqrt(2*TotalBoostPotential*(VmaxP-VminP))+sqrt(2*TotalBoostPotential*(VmaxP-VminP)-4*(VminP-TotalEnergy)*(VmaxP-VminP)))/(2*(VminP-TotalEnergy)))^2"
        self.addComputeGlobal("Totalk0prime", Totalk0_expr_prime)
        self.addComputeGlobal("Totalk0doubleprime", Totalk0_expr_doubleprime)
        self.addComputeGlobal("Totalk0", "min(Totalk0prime, Totalk0doubleprime)")
        self.addComputeGlobal("TotalRefEnergy", "VminP+(VmaxP-VminP)/Totalk0")
        self.addComputeGlobal("TotalBoostMode", "2.0")
        self.beginIfBlock("Totalk0 >= 1.0")
        self.addComputeGlobal("TotalRefEnergy", "VmaxP")
        self.addComputeGlobal("Totalk0", "2*TotalBoostPotential*(VmaxP-VminP)/(TotalRefEnergy-TotalEnergy)^2")
        self.addComputeGlobal("TotalBoostMode", "1.0")
        self.endBlock()
        self.beginIfBlock("TotalEnergy < TotalRefEnergy")
        self.addComputeGlobal("TotalForceScalingFactor", "1-Totalk0*(TotalRefEnergy-TotalEnergy)/(VmaxP-VminP)")
        self.addComputeGlobal("TotalEnergy", "TotalEnergy+TotalBoostPotential")
        self.endBlock()
        self.beginIfBlock("TotalEnergy >= TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.0")
        self.addComputeGlobal("Totalk0", "0.0")
        self.addComputeGlobal("TotalForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("VminP", "min(TotalEnergy, VminP)")
        self.addComputeGlobal("VmaxP", "max(TotalEnergy, VmaxP)")

        self.addComputeGlobal("vscale", "exp(-dt*collision_rate)")
        self.addComputeGlobal("fscale", "(1-vscale)/collision_rate")
        self.addComputeGlobal("noisescale", "sqrt(thermal_energy*(1-vscale*vscale))")
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("v", "vscale*v + noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("v", "v + fscale*f2*TotalForceScalingFactor*DihedralForceScalingFactor/m")
        self.addComputePerDof("v", "v + fscale*f0*TotalForceScalingFactor/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-oldx)/dt")
