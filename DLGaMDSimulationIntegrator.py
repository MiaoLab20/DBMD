import pandas as pd
import numpy as np
import sys, os

restartFile = "gamd-restart.dat"
isExist = os.path.exists(restartFile)
if isExist:
    column_names = ["Parameters", "Values"]
    gamdRestart = pd.read_csv("gamd-restart.dat", names=column_names, header=0, delimiter="\t")
    boost_parameters = gamdRestart.Values.to_list()
    boost_parameters = np.array(boost_parameters)

    nbsteps = boost_parameters[0]
    min_dihedral, max_dihedral = boost_parameters[1] * 4.184, boost_parameters[2] * 4.184
    min_total, max_total = boost_parameters[6] * 4.184, boost_parameters[7] * 4.184
    final_k0D, final_k0P = boost_parameters[5], boost_parameters[10]
else:
    nbsteps = 0
    min_dihedral, max_dihedral = 0 * 4.184, 0 * 4.184
    min_total, max_total = 0 * 4.184, 0 * 4.184
    final_k0D, final_k0P = 0, 0

from math import sqrt
from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from utils import *
from simParams import *
"""
OpenMM Custom Integrator Manual: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
"""
class conventionalMDIntegrator(CustomIntegrator):
    def __init__(self, dt=0.002*picoseconds, temperature=temperature*kelvins):
        CustomIntegrator.__init__(self, dt)

        self.collision_rate = 1/picoseconds
        self.kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        self.temperature = temperature
        self.thermal_energy = self.kB * self.temperature

        self.addGlobalVariable("thermal_energy", self.thermal_energy)
        self.addGlobalVariable("collision_rate", self.collision_rate)
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()

        self.addComputeGlobal("vscale", "exp(-dt*collision_rate)")
        self.addComputeGlobal("fscale", "(1-vscale)/collision_rate")
        self.addComputeGlobal("noisescale", "sqrt(thermal_energy*(1-vscale*vscale))")
        self.addComputePerDof("oldx", "x")
        self.addComputePerDof("v", "vscale*v + fscale*f/m + noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-oldx)/dt")

class DualDeepLearningGaMDEquilibration(CustomIntegrator):
    def __init__(self, dt=0.002*picoseconds, temperature=temperature*kelvins):
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
        self.addGlobalVariable("BoostedDihedralEnergy", 0.0)
        self.addGlobalVariable("VminD", min_dihedral)
        self.addGlobalVariable("VmaxD", max_dihedral)
        self.addGlobalVariable("DihedralBoostPotential", 0.0)
        self.addGlobalVariable("Dihedralk0", final_k0D)
        self.addGlobalVariable("DihedralRefEnergy", 0.0)
        self.addGlobalVariable("DihedralForceScalingFactor", 1.0)
        self.addGlobalVariable("TotalEnergy", 0.0)
        self.addGlobalVariable("BoostedTotalEnergy", 0.0)
        self.addGlobalVariable("VminP", min_total)
        self.addGlobalVariable("VmaxP", max_total)
        self.addGlobalVariable("TotalBoostPotential", 0.0)
        self.addGlobalVariable("Totalk0", final_k0P)
        self.addGlobalVariable("TotalRefEnergy", 0.0)
        self.addGlobalVariable("TotalForceScalingFactor", 1.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()

        self.addComputeGlobal("DihedralEnergy", "energy2")
        self.addComputeGlobal("DihedralRefEnergy", "VminD+(VmaxD-VminD)/Dihedralk0")
        self.beginIfBlock(f"DihedralRefEnergy > VmaxD+{refED_factor}*abs(VmaxD)")
        self.addComputeGlobal("DihedralRefEnergy", "VmaxD")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy < DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "(0.5)*Dihedralk0*(DihedralRefEnergy-DihedralEnergy)^2/(VmaxD-VminD)")
        self.addComputeGlobal("DihedralForceScalingFactor", "1-Dihedralk0*(DihedralRefEnergy-DihedralEnergy)/(VmaxD-VminD)")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy >= DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.0")
        self.addComputeGlobal("DihedralForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedDihedralEnergy", "DihedralEnergy+DihedralBoostPotential")
        self.addComputeGlobal("VminD", "min(BoostedDihedralEnergy, VminD)")
        self.addComputeGlobal("VmaxD", "max(BoostedDihedralEnergy, VmaxD)")
    
        self.addComputeGlobal("TotalEnergy", "energy+DihedralBoostPotential")
        self.addComputeGlobal("TotalRefEnergy", "VminP+(VmaxP-VminP)/Totalk0")
        self.beginIfBlock(f"TotalRefEnergy > VmaxP+{refEP_factor}*abs(VmaxP)")
        self.addComputeGlobal("TotalRefEnergy", "VmaxP")
        self.endBlock()
        self.beginIfBlock("TotalEnergy < TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "(0.5)*Totalk0*(TotalRefEnergy-TotalEnergy)^2/(VmaxP-VminP)")
        self.addComputeGlobal("TotalForceScalingFactor", "1-Totalk0*(TotalRefEnergy-TotalEnergy)/(VmaxP-VminP)")
        self.endBlock()
        self.beginIfBlock("TotalEnergy >= TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.0")
        self.addComputeGlobal("TotalForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedTotalEnergy", "TotalEnergy+TotalBoostPotential")
        self.addComputeGlobal("VminP", "min(BoostedTotalEnergy, VminP)")
        self.addComputeGlobal("VmaxP", "max(BoostedTotalEnergy, VmaxP)")

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

class DualDeepLearningGaMDProduction(CustomIntegrator):
    def __init__(self, dt=0.002 * picoseconds, temperature=temperature*kelvins):
        CustomIntegrator.__init__(self, dt)

        self.collision_rate = 1 / picosecond
        self.kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        self.temperature = temperature
        self.thermal_energy = self.kB * self.temperature

        self.addGlobalVariable("thermal_energy", self.thermal_energy)
        self.addGlobalVariable("collision_rate", self.collision_rate)
        self.addGlobalVariable("vscale", 0.0)
        self.addGlobalVariable("fscale", 0.0)
        self.addGlobalVariable("noisescale", 0.0)
        self.addGlobalVariable("DihedralEnergy", 0.0)
        self.addGlobalVariable("BoostedDihedralEnergy", 0.0)
        self.addGlobalVariable("VminD", min_dihedral)
        self.addGlobalVariable("VmaxD", max_dihedral)
        self.addGlobalVariable("DihedralBoostPotential", 0.0)
        self.addGlobalVariable("Dihedralk0", final_k0D)
        self.addGlobalVariable("DihedralRefEnergy", 0.0)
        self.addGlobalVariable("DihedralForceScalingFactor", 1.0)
        self.addGlobalVariable("TotalEnergy", 0.0)
        self.addGlobalVariable("BoostedTotalEnergy", 0.0)
        self.addGlobalVariable("VminP", min_total)
        self.addGlobalVariable("VmaxP", max_total)
        self.addGlobalVariable("TotalBoostPotential", 0.0)
        self.addGlobalVariable("Totalk0", final_k0P)
        self.addGlobalVariable("TotalRefEnergy", 0.0)
        self.addGlobalVariable("TotalForceScalingFactor", 1.0)
        self.addPerDofVariable("oldx", 0.0)
        self.addUpdateContextState()

        self.addComputeGlobal("DihedralEnergy", "energy2")
        self.addComputeGlobal("DihedralRefEnergy", "VminD+(VmaxD-VminD)/Dihedralk0")
        self.beginIfBlock(f"DihedralRefEnergy > VmaxD+{refED_factor}*abs(VmaxD)")
        self.addComputeGlobal("DihedralRefEnergy", "VmaxD")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy < DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "(0.5)*Dihedralk0*(DihedralRefEnergy-DihedralEnergy)^2/(VmaxD-VminD)")
        self.addComputeGlobal("DihedralForceScalingFactor", "1-Dihedralk0*(DihedralRefEnergy-DihedralEnergy)/(VmaxD-VminD)")
        self.endBlock()
        self.beginIfBlock("DihedralEnergy >= DihedralRefEnergy")
        self.addComputeGlobal("DihedralBoostPotential", "0.0")
        self.addComputeGlobal("DihedralForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedDihedralEnergy", "DihedralEnergy+DihedralBoostPotential")

        self.addComputeGlobal("TotalEnergy", "energy+DihedralBoostPotential")
        self.addComputeGlobal("TotalRefEnergy", "VminP+(VmaxP-VminP)/Totalk0")
        self.beginIfBlock(f"TotalRefEnergy > VmaxP+{refEP_factor}*abs(VmaxP)")
        self.addComputeGlobal("TotalRefEnergy", "VmaxP")
        self.endBlock()
        self.beginIfBlock("TotalEnergy < TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "(0.5)*Totalk0*(TotalRefEnergy-TotalEnergy)^2/(VmaxP-VminP)")
        self.addComputeGlobal("TotalForceScalingFactor", "1-Totalk0*(TotalRefEnergy-TotalEnergy)/(VmaxP-VminP)")
        self.endBlock()
        self.beginIfBlock("TotalEnergy >= TotalRefEnergy")
        self.addComputeGlobal("TotalBoostPotential", "0.0")
        self.addComputeGlobal("TotalForceScalingFactor", "1.0")
        self.endBlock()
        self.addComputeGlobal("BoostedTotalEnergy", "TotalEnergy+TotalBoostPotential")

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
