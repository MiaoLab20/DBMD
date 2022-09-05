import pandas as pd
import numpy as np
import sys, os

column_names = ["Parameters", "Values"]
gamdRestart = pd.read_csv("gamd-restart.dat", names=column_names, header=0, delimiter="\t")
boost_parameters = gamdRestart.Values.to_list()
boost_parameters = np.array(boost_parameters)

nbsteps = boost_parameters[0]
# dihedralEnergy, totalEnergy = boost_parameters[1] * 4.184, boost_parameters[6] * 4.184
min_dihedral, max_dihedral = boost_parameters[1] * 4.184, boost_parameters[2] * 4.184
min_total, max_total = boost_parameters[6] * 4.184, boost_parameters[7] * 4.184
final_k0D, final_k0P = boost_parameters[4], boost_parameters[9]

from math import sqrt
from utils import *
from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from utils import *
import sys, os
"""
OpenMM Custom Integrator Manual: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
"""
class DualDeepLearningGaMDProduction(CustomIntegrator):
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
        self.addGlobalVariable("DihedralBoostPotential", 0.0)
        self.addGlobalVariable("Dihedralk0prime", final_k0D)
        self.addGlobalVariable("Dihedralk0doubleprime", final_k0D)
        self.addGlobalVariable("Dihedralk0", final_k0D)
        self.addGlobalVariable("DihedralRefEnergy", 0.0)
        self.addGlobalVariable("DihedralForceScalingFactor", 1.0)
        self.addGlobalVariable("DihedralBoostMode", 0.0)
        self.addGlobalVariable("TotalEnergy", 0.0)
        self.addGlobalVariable("VminP", min_total)
        self.addGlobalVariable("VmaxP", max_total)
        self.addGlobalVariable("TotalBoostPotential", 0.0)
        self.addGlobalVariable("Totalk0prime", final_k0P)
        self.addGlobalVariable("Totalk0doubleprime", final_k0P)
        self.addGlobalVariable("Totalk0", final_k0P)
        self.addGlobalVariable("TotalRefEnergy", 0.0)
        self.addGlobalVariable("TotalForceScalingFactor", 1.0)
        self.addGlobalVariable("TotalBoostMode", 0.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()

        self.addComputeGlobal("DihedralEnergy", "energy2")
        self.setGlobalVariableByName("Dihedralk0", final_k0D)
        # self.addComputeGlobal("DihedralRefEnergy", "VminD+(VmaxD-VminD)/Dihedralk0")
        # self.addComputeGlobal("DihedralBoostMode", "2.0")
        self.addComputeGlobal("DihedralRefEnergy", "VmaxD")
        self.addComputeGlobal("DihedralBoostMode", "1.0")
        self.beginIfBlock("DihedralEnergy < DihedralRefEnergy")
        self.addComputeGlobal("DihedralForceScalingFactor", "1-Dihedralk0*(DihedralRefEnergy-DihedralEnergy)/(VmaxD-VminD)")
        self.addComputeGlobal("DihedralBoostPotential", "(1/2)*Dihedralk0*(DihedralRefEnergy-DihedralEnergy)^2/(VmaxD-VminD)")
        self.addComputeGlobal("DihedralEnergy", "DihedralEnergy+DihedralBoostPotential")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy >= DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.0")
        self.addComputeGlobal("DihedralForceScalingFactor", "1.0")
        self.endBlock()
        # self.addComputeGlobal("VminD", "min(DihedralEnergy, VminD)")
        # self.addComputeGlobal("VmaxD", "max(DihedralEnergy, VmaxD)")
    
        self.addComputeGlobal("TotalEnergy", "energy0+DihedralBoostPotential")
        self.setGlobalVariableByName("Totalk0", final_k0P)
        # self.addComputeGlobal("TotalRefEnergy", "VminP+(VmaxP-VminP)/Totalk0")
        # self.addComputeGlobal("TotalBoostMode", "2.0")
        self.addComputeGlobal("TotalRefEnergy", "VmaxP")
        self.addComputeGlobal("TotalBoostMode", "1.0")
        self.beginIfBlock("TotalEnergy < TotalRefEnergy")
        self.addComputeGlobal("TotalForceScalingFactor", "1-Totalk0*(TotalRefEnergy-TotalEnergy)/(VmaxP-VminP)")
        self.addComputeGlobal("TotalBoostPotential", "(1/2)*Totalk0*(TotalRefEnergy-TotalEnergy)^2/(VmaxP-VminP)")
        self.addComputeGlobal("TotalEnergy", "TotalEnergy+TotalBoostPotential")
        self.endBlock()
        self.beginIfBlock("TotalEnergy >= TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.0")
        self.addComputeGlobal("TotalForceScalingFactor", "1.0")
        self.endBlock()
        # self.addComputeGlobal("VminP", "min(TotalEnergy, VminP)")
        # self.addComputeGlobal("VmaxP", "max(TotalEnergy, VmaxP)")

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
