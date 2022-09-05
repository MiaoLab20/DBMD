from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from DLGaMDProductionIntegrator2 import *
from utils import *
"""
OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
"""
prmtop = AmberPrmtopFile("gcaa_unfold.top")
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=8*angstrom, constraints=HBonds)
# system = prmtop.createSystem(implicitSolvent=GBn2, implicitSolventKappa=1.0/nanometer, soluteDielectric=1.0, solventDielectric=78.5)
set_dihedral_group(system)

integrator = DualDeepLearningGaMDProduction(0.002*picoseconds, 300*kelvins)
isExist = os.path.exists("gamd.mdout")
if isExist:
    os.remove("gamd.mdout")
simulation = Simulation(prmtop.topology, system, integrator)
simulation.loadState("equil.rst")
integrator.setGlobalVariableByName("Dihedralk0", final_k0D)
integrator.setGlobalVariableByName("Totalk0", final_k0P)
simulation.reporters.append(DCDReporter("gamd.dcd", 500))
simulation.reporters.append(ExpandedStateDataReporter(system, "gamd.mdout", 500, step=True, temperature=True,
                                                    brokenOutForceEnergies=True, potentialEnergy=True,
                                                    totalEnergy=True, density=True, separator="\t",
                                                    speed=True, remainingTime=True, totalSteps=750000000))
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

for step in np.arange(0, 1500000):
    simulation.step(500)
    simulation.saveState("gamd.rst")
    gamdLog = open("gamd.log", "a")
    gamdLog.write(str(500) + "\t" + str((step+1)*500) + "\t"
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

    isExist = os.path.exists("gamd-restart.dat")
    if isExist:
        os.remove("gamd-restart.dat")
    gamdRestart = open("gamd-restart.dat", "w")
    gamdRestart.write("#Parameters\tValues(kcal/mol)\n")
    gamdRestart.write("(0)Steps:\t" + str((step+1)*500) + "\n")
    gamdRestart.write("(1)Boosted VfinalD:\t" + str(integrator.getGlobalVariableByName("DihedralEnergy") / 4.184) + "\n")
    gamdRestart.write("(2)Boosted VminD:\t" + str(integrator.getGlobalVariableByName("VminD") / 4.184) + "\n")
    gamdRestart.write("(3)Boosted VmaxD:\t" + str(integrator.getGlobalVariableByName("VmaxD") / 4.184) + "\n")
    gamdRestart.write("(4)Final k0D:\t" + str(integrator.getGlobalVariableByName("Dihedralk0")) + "\n")
    gamdRestart.write("(5)Final DihedralBoost:\t" + str(integrator.getGlobalVariableByName("DihedralBoostPotential") / 4.184) + "\n")
    gamdRestart.write("(6)Boosted VfinalP:\t" + str(integrator.getGlobalVariableByName("TotalEnergy") / 4.184) + "\n")
    gamdRestart.write("(7)Boosted VminP:\t" + str(integrator.getGlobalVariableByName("VminP") / 4.184) + "\n")
    gamdRestart.write("(8)Boosted VmaxP:\t" + str(integrator.getGlobalVariableByName("VmaxP") / 4.184) + "\n")
    gamdRestart.write("(9)Final k0P:\t" + str(integrator.getGlobalVariableByName("Totalk0")) + "\n")
    gamdRestart.write("(10)Final TotalBoost:\t" + str(integrator.getGlobalVariableByName("TotalBoostPotential") / 4.184) + "\n")
    gamdRestart.close()
