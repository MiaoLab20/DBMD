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
import seaborn as sns
import matplotlib.pyplot as plt

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
from simParams import *

tfpd = tfp.distributions
tfpl = tfp.layers

from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *

for cycle in np.arange(ncycebstart, ncycebend):
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
    if cycle == 0: df = pd.read_csv("equil-prep.mdout", names=column_names, header=0, delimiter="\t")
    else: df = pd.read_csv("equil.mdout", names=column_names, header=0, delimiter="\t")

    steps = df.Steps.to_list()
    steps = np.array(steps[-10000:])
    steps = np.repeat(steps, 50, axis=-1)
    nbsteps = len(steps)

    dihedralEnergy = df.PeriodicTorsionForce.to_list()
    dihedralEnergy = np.array(dihedralEnergy[-10000:]) / 4.184
    dihedralEnergy = np.repeat(dihedralEnergy, 50, axis=-1)
    min_dihedral, max_dihedral = min(dihedralEnergy), max(dihedralEnergy)

    totalEnergy = df.PotentialEnergy.to_list()
    totalEnergy = np.array(totalEnergy[-10000:]) / 4.184
    totalEnergy = np.repeat(totalEnergy, 50, axis=-1)
    min_total, max_total = min(totalEnergy), max(totalEnergy)

    totalEnergies, totalBoosts, dihedralEnergies, dihedralBoosts = [], [], [], []

    for step in np.arange(0, len(steps)):
        totalEnergies.append(totalEnergy[step])
        totalfc0 = np.random.uniform(0, 1)
        while totalfc0 == 0: totalfc0 = np.random.uniform(0, 1)
        totalrefE = min_total + (max_total - min_total) / totalfc0
        totalboost = (1/2) * totalfc0 * (1/(max_total - min_total)) * (totalrefE - totalEnergy[step])**2
        if np.isnan(totalboost) == True: totalboost = 0
        totalBoosts.append(totalboost)

        dihedralEnergies.append(dihedralEnergy[step])
        dihedralfc0 = np.random.uniform(0, 1)
        while dihedralfc0 == 0: dihedralfc0 = np.random.uniform(0, 1)
        dihedralrefE = min_dihedral + (max_dihedral - min_dihedral) / dihedralfc0
        dihedralboost = (1/2) * dihedralfc0 * (1/(max_dihedral - min_dihedral)) * (dihedralrefE - dihedralEnergy[step])**2
        if np.isnan(dihedralboost) == True: dihedralboost = 0
        dihedralBoosts.append(dihedralboost)

    Xtot = np.array(totalEnergies)
    Ytot = np.array(totalBoosts)
    Xdih = np.array(dihedralEnergies)
    Ydih = np.array(dihedralBoosts)
    Ydual = np.add(Ytot,Ydih)
    anharmtot, anharmdih, anharmdual = anharm(Ytot), anharm(Ydih), anharm(Ydual)

    boostFolder = "equil-boosts-"+str(cycle)
    isExist = os.path.exists(boostFolder)
    if not isExist: os.makedirs(boostFolder)

    plt.figure(figsize=(9, 6))
    sns.kdeplot(Ytot, color="Blue", bw_method=0.25)
    sns.kdeplot(Ydih, color="Orange", bw_method=0.25)
    sns.kdeplot(Ydual, color="Green", bw_method=0.25)
    plt.legend(labels=["Total(γ="+str(round(anharmtot,3))+"; ∆V="+str(round(np.mean(Ytot), 2))+"±"+str(round(np.std(Ytot), 2))+")",
                       "Dihedral(γ="+str(round(anharmdih,3))+"; ∆V="+str(round(np.mean(Ydih), 2))+"±"+str(round(np.std(Ydih), 2))+")",
                       "Dual(γ="+str(round(anharmdual,3))+"; ∆V="+str(round(np.mean(Ydual), 2))+"±"+str(round(np.std(Ydual), 2))+")"], fontsize=15)
    plt.xlabel("∆V (kcal/mol)", fontsize=19, rotation=0)
    plt.ylabel("p(∆V)", fontsize=19, rotation=90)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.savefig(boostFolder + "/randomboosts.png")

    iter = 0
    while (anharmdual > 1e-2) and iter <= 2:
        Xtot_train, Xtot_test, Ytot_train, Ytot_test = train_test_split(Xtot, Ytot, test_size=0.2)
        Xdih_train, Xdih_test, Ydih_train, Ydih_test = train_test_split(Xdih, Ydih, test_size=0.2)

        modeltot = fullModel(nbsteps)
        modeldih = fullModel(nbsteps)

        totalFolder, dihedralFolder = "equil-cptot-"+str(cycle), "equil-cpdih-"+str(cycle)
        isExist = os.path.exists(totalFolder)
        if not isExist: os.makedirs(totalFolder)
        isExist = os.path.exists(dihedralFolder)
        if not isExist: os.makedirs(dihedralFolder)

        cptot_path = totalFolder + "/cptot-{epoch:04d}.ckpt"
        cpdih_path = dihedralFolder + "/cpdih-{epoch:04d}.ckpt"
        cptot_callback = ModelCheckpoint(filepath=cptot_path, save_weights_only=True, save_freq="epoch")
        cpdih_callback = ModelCheckpoint(filepath=cpdih_path, save_weights_only=True, save_freq="epoch")

        historytot = modeltot.fit(Xtot_train, Ytot_train, epochs=50, batch_size=100, verbose=0,
                              callbacks=[cptot_callback], validation_data=[Xtot_test, Ytot_test])
        historydih = modeldih.fit(Xdih_train, Ydih_train, epochs=50, batch_size=100, verbose=0,
                              callbacks=[cpdih_callback], validation_data=[Xdih_test, Ydih_test])

        Ytot = modeltot.predict(Xtot)
        Ytot = np.absolute(Ytot.flatten())
        Ydih = modeldih.predict(Xdih)
        Ydih = np.absolute(Ydih.flatten())
        Ydual = np.add(Ytot,Ydih)
        anharmtot, anharmdih, anharmdual = anharm(Ytot), anharm(Ydih), anharm(Ydual)

        plt.figure(figsize=(9, 6))
        sns.kdeplot(Ytot, color="Blue", bw_method=0.25)
        sns.kdeplot(Ydih, color="Orange", bw_method=0.25)
        sns.kdeplot(Ydual, color="Green", bw_method=0.25)
        plt.legend(labels=["Total(γ="+str(round(anharmtot,3))+"; ∆V="+str(round(np.mean(Ytot), 2))+"±"+str(round(np.std(Ytot), 2))+")",
                    "Dihedral(γ="+str(round(anharmdih, 3))+"; ∆V="+str(round(np.mean(Ydih), 2))+"±"+str(round(np.std(Ydih), 2))+")",
                    "Dual(γ="+str(round(anharmdual, 3))+"; ∆V="+str(round(np.mean(Ydual), 2))+"±"+str(round(np.std(Ydual), 2))+")"], fontsize=15)
        plt.xlabel("∆V (kcal/mol)", fontsize=19, rotation=0)
        plt.ylabel("p(∆V)", fontsize=19, rotation=90)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.savefig(boostFolder + "/DLBoosts" + str(iter + 1) + ".png")

        totalfc0s, dihedralfc0s, totalfscales, dihedralfscales = [], [], [], []
        for i in np.arange(0, len(Ydih), 100):
            dVD, VminD, VmaxD, VD = Ydih[i], min_dihedral, max_dihedral, Xdih[i]
            k0D = ((sqrt(2*dVD*(VmaxD-VminD))-sqrt(2*dVD*(VmaxD-VminD)-4*(VminD-VD)*(VmaxD-VminD)))/(2*(VminD-VD)))**2
            dihedralrefE = VminD + (VmaxD - VminD) / k0D
            if (k0D > 1.0) or (dihedralrefE > VmaxD+refED_factor*np.abs(VmaxD)):
                dihedralrefE = VmaxD
                k0D = (2*dVD*(VmaxD-VminD)) / (dihedralrefE-VD)**2
                k0D = min(1.0, k0D)
            kD = 1 - (k0D/(VmaxD-VminD)) * (dihedralrefE-VD)
            dihedralfc0s.append(k0D)
            dihedralfscales.append(kD)

            dVP, VminP, VmaxP, VP = Ytot[i], min_total, max_total, Xtot[i]
            k0P = ((sqrt(2*dVP*(VmaxP-VminP))-sqrt(2*dVP*(VmaxP-VminP)-4*(VminP-VP)*(VmaxP-VminP)))/(2*(VminP-VP)))**2
            totalrefE = VminP + (VmaxP - VminP) / k0P
            if (k0P > 1.0) or (totalrefE > VmaxP+refEP_factor*np.abs(VmaxP)):
                totalrefE = VmaxP
                k0P = (2*dVP*(VmaxP-VminP)) / (totalrefE-VP)**2
                k0P = min(1.0, k0P)
            kP = 1 - (k0P/(VmaxP-VminP)) * (totalrefE-VP)
            totalfc0s.append(k0P)
            totalfscales.append(kP)

        totalfc0s = np.asarray([totalfc0s]).flatten()
        dihedralfc0s = np.asarray([dihedralfc0s]).flatten()
        plt.figure(figsize=(9, 6))
        plt.scatter(np.arange(0, len(totalfc0s)), totalfc0s, label="Total k0P")
        plt.scatter(np.arange(0, len(dihedralfc0s)), dihedralfc0s, label="Dihedral k0D")
        plt.ylim(0, 1.5)
        plt.xlabel("Steps", fontsize=19, rotation=0)
        plt.ylabel("k0P / k0D", fontsize=19, rotation=90)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=15)
        plt.savefig(boostFolder + "/efconstants0" + str(iter + 1) + ".png")

        restartFile = "gamd-restart.dat"
        isExist = os.path.exists(restartFile)
        if isExist: os.remove(restartFile)
        gamdRestart = open(restartFile, "w")
        gamdRestart.write("#Parameters\tValues(kcal/mol)\n")
        gamdRestart.write("(0)Trained Steps:\t" + str(nbsteps) + "\n")
        gamdRestart.write("(1)Boosted VminD:\t" + str(min(dihedralEnergy)) + "\n")
        gamdRestart.write("(2)Boosted VmaxD:\t" + str(max(dihedralEnergy)) + "\n")
        gamdRestart.write("(3)Boosted VfinalD:\t" + str(Xdih[-1]) + "\n")
        gamdRestart.write("(4)Average k0D:\t" + str(np.mean([k0D for k0D in dihedralfc0s if np.isnan(k0D) == False])) + "\n")
        gamdRestart.write("(5)Final k0D:\t" + str(dihedralfc0s[-1]) + "\n")
        gamdRestart.write("(6)Boosted VminP:\t" + str(min(totalEnergy)) + "\n")
        gamdRestart.write("(7)Boosted VmaxP:\t" + str(max(totalEnergy)) + "\n")
        gamdRestart.write("(8)Boosted VfinalP:\t" + str(Xtot[-1]) + "\n")
        gamdRestart.write("(9)Average k0P:\t" + str(np.mean([k0P for k0P in totalfc0s if np.isnan(k0P) == False])) + "\n")
        gamdRestart.write("(10)Final k0P:\t" + str(totalfc0s[-1]) + "\n")
        gamdRestart.close()

        iter += 1

    os.system("scp gamd-restart.dat " + "equil-"+str(cycle)+".restart.dat")

    from DLGaMDSimulationIntegrator import *
    """
    OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
    """
    prmtop = AmberPrmtopFile(parmFile)
    if simType == "explicit": system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=nbCutoff*angstrom, constraints=HBonds)
    else: system = prmtop.createSystem(implicitSolvent=GBn2, implicitSolventKappa=1.0/nanometer, soluteDielectric=1.0, solventDielectric=78.5)
    set_dihedral_group(system)

    integrator = DualDeepLearningGaMDEquilibration(0.002*picoseconds, temperature*kelvins)
    isExist = os.path.exists("equil-"+str(cycle)+".mdout")
    if isExist: os.remove("equil-"+str(cycle)+".mdout")
    simulation = Simulation(prmtop.topology, system, integrator)

    if cycle == 0: simulation.loadState("equil-prep.rst")
    else: simulation.loadState("equil.rst")

    column_names = ["Parameters", "Values"]
    gamdRestart = pd.read_csv("gamd-restart.dat", names=column_names, header=0, delimiter="\t")
    boost_parameters = gamdRestart.Values.to_list()
    boost_parameters = np.array(boost_parameters)
    min_dihedral, max_dihedral = boost_parameters[1] * 4.184, boost_parameters[2] * 4.184
    min_total, max_total = boost_parameters[6] * 4.184, boost_parameters[7] * 4.184
    final_k0D, final_k0P = boost_parameters[5], boost_parameters[10]

    integrator.setGlobalVariableByName("VminD", min_dihedral)
    integrator.setGlobalVariableByName("VmaxD", max_dihedral)
    integrator.setGlobalVariableByName("Dihedralk0", final_k0D)
    integrator.setGlobalVariableByName("VminP", min_total)
    integrator.setGlobalVariableByName("VmaxP", max_total)
    integrator.setGlobalVariableByName("Totalk0", final_k0P)

    simulation.reporters.append(DCDReporter("equil-"+str(cycle)+".dcd", ebRestartFreq))
    simulation.reporters.append(ExpandedStateDataReporter(system, "equil-"+str(cycle)+".mdout", ebRestartFreq,
                                                          step=True, temperature=True,
                                                          brokenOutForceEnergies=True, potentialEnergy=True,
                                                          totalEnergy=True, density=True, separator="\t",
                                                          speed=True, remainingTime=True, totalSteps=ntebpercyc))
    logFile = "equil-"+str(cycle)+".log"
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

    for step in np.arange(0, ntebpercyc / ebRestartFreq):
        simulation.step(ebRestartFreq)
        simulation.saveState("equil-"+str(cycle)+".rst")
        gamdLog = open(logFile, "a")
        gamdLog.write(str(ebRestartFreq) + "\t" + str(int((step + 1) * ebRestartFreq)) + "\t"
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
        gamdRestart.write("(0)Steps:\t" + str(int((step + 1)*ebRestartFreq)) + "\n")
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

    os.system("scp equil-"+str(cycle)+".mdout equil.mdout")
    os.system("scp gamd-restart.dat equil-"+str(cycle)+".restart.dat")
    os.system("scp equil-"+str(cycle)+".rst equil.rst")
