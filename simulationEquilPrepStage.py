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

import pandas as pd, numpy as np, os, sys
from utils import *
from simParams import *
from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *

for cycle in np.arange(ncycebprepstart, ncycebprepend):
    if simType == "protein.implicit":
        column_names = ["Steps", "PotentialEnergy", "TotalEnergy", "Temperature", "Density", "Speed",
                    "TimeRemaining", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                    "CMAPTorsionForce", "NonbondedForce", "CustomGBForce", "CMMotionRemover"]
    elif simType == "RNA.implicit":
        column_names = ["Steps", "PotentialEnergy", "TotalEnergy", "Temperature", "Density", "Speed",
                    "TimeRemaining", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                    "NonbondedForce", "CustomGBForce", "CMMotionRemover"]
    else: # simType == "explicit"
        column_names = ["Steps", "PotentialEnergy", "TotalEnergy", "Temperature", "Density", "Speed",
                    "TimeRemaining", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                    "NonbondedForce", "CMMotionRemover"]
    if cycle == 0: df = pd.read_csv("cmd.mdout", names=column_names, header=0, delimiter="\t")
    else: df = pd.read_csv("equil-prep.mdout", names=column_names, header=0, delimiter="\t")

    steps = df.Steps.to_list()
    steps = np.array(steps[-10000:])
    nbsteps = len(steps)

    dihedralEnergy = df.PeriodicTorsionForce.to_list()
    dihedralEnergy = np.array(dihedralEnergy[-10000:]) / 4.184
    min_dihedral, max_dihedral = min(dihedralEnergy), max(dihedralEnergy)

    totalEnergy = df.PotentialEnergy.to_list()
    totalEnergy = np.array(totalEnergy[-10000:]) / 4.184
    min_total, max_total = min(totalEnergy), max(totalEnergy)

    restartFile = "gamd-restart.dat"
    isExist = os.path.exists(restartFile)
    if isExist: os.remove(restartFile)
    gamdRestart = open(restartFile, "w")
    gamdRestart.write("#Parameters\tValues(kcal/mol)\n")
    gamdRestart.write("(0)Trained Steps:\t" + str(nbsteps) + "\n")
    gamdRestart.write("(1)Boosted VminD:\t" + str(min(dihedralEnergy)) + "\n")
    gamdRestart.write("(2)Boosted VmaxD:\t" + str(max(dihedralEnergy)) + "\n")
    gamdRestart.write("(3)Boosted VfinalD:\t" + str(dihedralEnergy[-1]) + "\n")
    gamdRestart.write("(4)Average k0D:\t" + str(1.0) + "\n")
    gamdRestart.write("(5)Final k0D:\t" + str(1.0) + "\n")
    gamdRestart.write("(6)Boosted VminP:\t" + str(min(totalEnergy)) + "\n")
    gamdRestart.write("(7)Boosted VmaxP:\t" + str(max(totalEnergy)) + "\n")
    gamdRestart.write("(8)Boosted VfinalP:\t" + str(totalEnergy[-1]) + "\n")
    gamdRestart.write("(9)Average k0P:\t" + str(1.0) + "\n")
    gamdRestart.write("(10)Final k0P:\t" + str(1.0) + "\n")
    gamdRestart.close()

    if cycle == 0: os.system("scp -v gamd-restart.dat cmd-restart.dat")
    else: os.system("scp -v gamd-restart.dat " + "equil-prep-"+str(cycle)+".restart.dat")

    from DLGaMDSimulationIntegrator import *
    """
    OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
    """
    prmtop = AmberPrmtopFile(parmFile)
    if simType == "explicit": system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=nbCutoff*angstrom, constraints=HBonds)
    else: system = prmtop.createSystem(implicitSolvent=GBn2, implicitSolventKappa=1.0/nanometer, soluteDielectric=1.0, solventDielectric=78.5)
    set_dihedral_group(system)

    integrator = DualDeepLearningGaMDEquilibration(0.002*picoseconds, temperature*kelvins)
    isExist = os.path.exists("equil-prep-"+str(cycle)+".mdout")
    if isExist: os.remove("equil-prep-"+str(cycle)+".mdout")
    simulation = Simulation(prmtop.topology, system, integrator)

    if cycle == 0: simulation.loadState("cmd.rst")
    else: simulation.loadState("equil-prep.rst")

    column_names = ["Parameters", "Values"]
    gamdRestart = pd.read_csv("gamd-restart.dat", names=column_names, header=0, delimiter="\t")
    boost_parameters = gamdRestart.Values.to_list()
    boost_parameters = np.array(boost_parameters)
    min_dihedral, max_dihedral = boost_parameters[1] * 4.184, boost_parameters[2] * 4.184
    min_total, max_total = boost_parameters[6] * 4.184, boost_parameters[7] * 4.184
    if simType == "explicit": final_k0D, final_k0P = 1.0, 1.0
    else: final_k0D, final_k0P = 1.0, 0.05

    integrator.setGlobalVariableByName("VminD", min_dihedral)
    integrator.setGlobalVariableByName("VmaxD", max_dihedral)
    integrator.setGlobalVariableByName("Dihedralk0", final_k0D)
    integrator.setGlobalVariableByName("VminP", min_total)
    integrator.setGlobalVariableByName("VmaxP", max_total)
    integrator.setGlobalVariableByName("Totalk0", final_k0P)

    simulation.reporters.append(DCDReporter("equil-prep-"+str(cycle)+".dcd", ebprepRestartFreq))
    simulation.reporters.append(ExpandedStateDataReporter(system, "equil-prep-"+str(cycle)+".mdout", ebprepRestartFreq,
                                                          step=True, temperature=True,
                                                          brokenOutForceEnergies=True, potentialEnergy=True,
                                                          totalEnergy=True, density=True, separator="\t",
                                                          speed=True, remainingTime=True, totalSteps=ntebpreppercyc))
    logFile = "equil-prep-"+str(cycle)+".log"
    isExist = os.path.exists(logFile)
    if isExist: os.remove(logFile)
    gamdLog = open(logFile, "w")
    gamdLog.write("# Gaussian accelerated Molecular Dynamics log file\n")
    gamdLog.write("# All energy terms are stored in unit of kcal/mol\n")
    gamdLog.write("# ntwx,total_nstep,Total-Energy,Dihedral-Energy,Total-Force-Weight,Dihedral-Force-Weight,"
                  "Total-Boost,Dihedral-Boost,Total-Harmonic-Force-Constant,Dihedral-Harmonic-Force-Constant,"
                  "Minimum-Total-Energy,Maximum-Total-Energy,Minimum-Dihedral-Energy,Maximum-Dihedral-Energy,"
                  "Total-Reference-Energy,Dihedral-Reference-Energy\n")
    gamdLog.close()

    for step in np.arange(0, ntebpreppercyc / ebprepRestartFreq):
        simulation.step(ebprepRestartFreq)
        simulation.saveState("equil-prep-"+str(cycle)+".rst")
        gamdLog = open(logFile, "a")
        gamdLog.write(str(ebprepRestartFreq) + "\t" + str(int((step + 1) * ebprepRestartFreq)) + "\t"
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
                      + str(integrator.getGlobalVariableByName("TotalRefEnergy") / 4.184) + "\t"
                      + str(integrator.getGlobalVariableByName("DihedralRefEnergy") / 4.184) + "\n")
        gamdLog.close()

        isExist = os.path.exists("gamd-restart.dat")
        if isExist: os.remove("gamd-restart.dat")
        gamdRestart = open("gamd-restart.dat", "w")
        gamdRestart.write("#Parameters\tValues(kcal/mol)\n")
        gamdRestart.write("(0)Steps:\t" + str(int((step + 1)*ebprepRestartFreq)) + "\n")
        gamdRestart.write("(1)Boosted VminD:\t" + str(integrator.getGlobalVariableByName("VminD") / 4.184) + "\n")
        gamdRestart.write("(2)Boosted VmaxD:\t" + str(integrator.getGlobalVariableByName("VmaxD") / 4.184) + "\n")
        gamdRestart.write("(3)DihedralRefEnergy:\t" + str(integrator.getGlobalVariableByName("DihedralRefEnergy") / 4.184) + "\n")
        gamdRestart.write("(4)Final DihedralBoost:\t" + str(integrator.getGlobalVariableByName("DihedralBoostPotential") / 4.184) + "\n")
        gamdRestart.write("(5)Final k0D:\t" + str(integrator.getGlobalVariableByName("Dihedralk0")) + "\n")
        gamdRestart.write("(6)Boosted VminP:\t" + str(integrator.getGlobalVariableByName("VminP") / 4.184) + "\n")
        gamdRestart.write("(7)Boosted VmaxP:\t" + str(integrator.getGlobalVariableByName("VmaxP") / 4.184) + "\n")
        gamdRestart.write("(8)TotalRefEnergy:\t" + str(integrator.getGlobalVariableByName("TotalRefEnergy") / 4.184) + "\n")
        gamdRestart.write("(9)Final TotalBoost:\t" + str(integrator.getGlobalVariableByName("TotalBoostPotential") / 4.184) + "\n")
        gamdRestart.write("(10)Final k0P:\t" + str(integrator.getGlobalVariableByName("Totalk0")) + "\n")
        gamdRestart.close()

    os.system("scp equil-prep-"+str(cycle)+".mdout equil-prep.mdout")
    os.system("scp gamd-restart.dat equil-prep-"+str(cycle)+".restart.dat")
    os.system("scp equil-prep-"+str(cycle)+".rst equil-prep.rst")
