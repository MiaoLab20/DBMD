from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from DLGaMDProductionIntegrator2 import *
from utils import *
"""
OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
"""
prmtop = AmberPrmtopFile("dip.top")
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=8*angstrom, constraints=HBonds)
set_dihedral_group(system)

integrator = DualDeepLearningGaMDProduction(0.002*picoseconds, 300*kelvins)
isExist = os.path.exists("gamd.mdout")
if isExist:
    os.remove("gamd.mdout")
simulation = Simulation(prmtop.topology, system, integrator)
simulation.loadState("equil.rst")
integrator.setGlobalVariableByName("Dihedralk0", final_k0D)
integrator.setGlobalVariableByName("Totalk0", final_k0P)
simulation.reporters.append(DCDReporter("gamd.dcd", 10))
simulation.reporters.append(ExpandedStateDataReporter(system, "gamd.mdout", 10, step=True, temperature=True,
                                                    brokenOutForceEnergies=True, potentialEnergy=True,
                                                    totalEnergy=True, density=True, separator="\t",
                                                    speed=True, remainingTime=True, totalSteps=30000000))
isExist = os.path.exists("gamd.log")
if isExist:
    os.remove("gamd.log")
gamdLog = open("gamd.log", "w")
gamdLog.write("# Gaussian accelerated Molecular Dynamics log file\n")
gamdLog.write("# All energy terms are stored in unit of kcal/mol\n")
gamdLog.write("# ntwx,total_nstep,Total-Energy,Dihedral-Energy,Total-Force-Weight,Dihedral-Force-Weight,"
              "Total-Boost,Dihedral-Boost,Total-Harmonic-Force-Constant,Dihedral-Harmonic-Force-Constant,"
              "Minimum-Total-Energy,Maximum-Total-Energy,Minimum-Dihedral-Energy,Maximum-Dihedral-Energy"
              "Total-Boost-Mode,Dihedral-Boost-Mode\n")
gamdLog.close()

for step in np.arange(0, 3000000):
    simulation.step(10)
    simulation.saveState("gamd.rst")
    gamdLog = open("gamd.log", "a")
    gamdLog.write(str(10) + "\t" + str((step+1)*10) + "\t"
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
