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
import seaborn as sns
import matplotlib.pyplot as plt

column_names = ["Steps", "PotentialEnergy", "TotalEnergy", "Temperature", "Density", "Speed",
                "TimeRemaining", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                "NonbondedForce", "CMMotionRemover"]
# column_names = ["Steps", "PotentialEnergy", "TotalEnergy", "Temperature", "Density", "Speed",
#                 "TimeRemaining", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
#                 "NonbondedForce", "CustomGBForce", "CMMotionRemover"]
df = pd.read_csv("equil.mdout", names=column_names, header=0, delimiter="\t")

steps = df.Steps.to_list()
steps = np.array(steps[-50000:])
steps = np.repeat(steps, 10, axis=-1)
nbsteps = len(steps)

dihedralEnergy = df.PeriodicTorsionForce.to_list()
dihedralEnergy = np.array(dihedralEnergy[-50000:]) / 4.184
dihedralEnergy = np.repeat(dihedralEnergy, 10, axis=-1)
min_dihedral, max_dihedral = min(dihedralEnergy), max(dihedralEnergy)
totalEnergy = df.PotentialEnergy.to_list()
totalEnergy = np.array(totalEnergy[-50000:]) / 4.184
totalEnergy = np.repeat(totalEnergy, 10, axis=-1)
totalEnergy = np.add(totalEnergy, dihedralEnergy)
min_total, max_total = min(totalEnergy), max(totalEnergy)

totalEnergies = []
totalBoosts = []
dihedralEnergies = []
dihedralBoosts = []

for step in np.arange(0, len(steps)):
    totalEnergies.append(totalEnergy[step])

    # totalfc0 = np.random.uniform(0, 1)
    # while totalfc0 == 0: totalfc0 = np.random.uniform(0, 1)

    totalfc0 = 1.0
    totalrefE = max_total + (max_total - min_total) * (1/totalfc0 - 1)
    totalboost = (1/2) * totalfc0 * (1/(max_total - min_total)) * (totalrefE - totalEnergy[step])**2
    if np.isnan(totalboost) == True: totalboost = 0
    totalBoosts.append(totalboost)

    dihedralEnergies.append(dihedralEnergy[step])

    # dihedralfc0 = np.random.uniform(0, 1)
    # while dihedralfc0 == 0: dihedralfc0 = np.random.uniform(0, 1)

    dihedralfc0 = 1.0
    dihedralrefE = max_dihedral + (max_dihedral - min_dihedral) * (1/dihedralfc0 - 1)
    dihedralboost = (1/2) * dihedralfc0 * (1/(max_dihedral - min_dihedral)) * (dihedralrefE - dihedralEnergy[step])**2
    if np.isnan(dihedralboost) == True: dihedralboost = 0
    dihedralBoosts.append(dihedralboost)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, Reduction
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp
from deeplearningmodel2 import *
from math import sqrt
from utils import *

tfpd = tfp.distributions
tfpl = tfp.layers

Xtot = np.array(totalEnergies)
Ytot = np.array(totalBoosts)
Xdih = np.array(dihedralEnergies)
Ydih = np.array(dihedralBoosts)
Ydual = np.add(Ytot,Ydih)
anharmtot, anharmdih, anharmdual = anharm(Ytot), anharm(Ydih), anharm(Ydual)

isExist = os.path.exists("equil-boosts")
if not isExist:
    os.makedirs("equil-boosts")

print("Iteration 0: ")
print("The anharmonicity of total boosts after iteration 0 is: " + str(anharmtot))
print("The minimum and maximum total boost potential after iteration 0 is: "
      + str(min(Ytot)) + " and " + str(max(Ytot)))
print("The anharmonicity of dihedral boosts after iteration 0 is: " + str(anharmdih))
print("The minimum and maximum dihedral boost potential after iteration 0 is: "
      + str(min(Ydih)) + " and " + str(max(Ydih)))
print("The anharmonicity of dual boosts after iteration 0 is: " + str(anharmdual))
print("The minimum and maximum dual boost potential after iteration 0 is: "
      + str(min(Ydual)) + " and " + str(max(Ydual)))
plt.figure(figsize=(9,6))
sns.kdeplot(Ytot, color="Blue")
sns.kdeplot(Ydih, color="Orange")
sns.kdeplot(Ydual, color="Green")
plt.legend(labels=["Total(" + str(round(anharmtot, 4)) + "; " + str(round(np.mean(Ytot), 3)) + "+/-" + str(round(np.std(Ytot), 3)) + ")",
                "Dihedral(" + str(round(anharmdih, 4)) + "; " + str(round(np.mean(Ydih), 3)) + "+/-" + str(round(np.std(Ydih), 3)) + ")",
                "Dual(" + str(round(anharmdual, 4)) + "; " + str(round(np.mean(Ydual), 3)) + "+/-" + str(round(np.std(Ydual), 3)) + ")"])
plt.xlabel("Boost (kcal/mol)", fontsize=16, rotation=0)
plt.ylabel("Frequency", fontsize=16, rotation=90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("equil-boosts/boostdistr0.png")

iter = 0
while (anharmdual > 1e-2) and iter <= 10:
    Xtot_train, Xtot_test, Ytot_train, Ytot_test = train_test_split(Xtot, Ytot, test_size=0.2)
    Xdih_train, Xdih_test, Ydih_train, Ydih_test = train_test_split(Xdih, Ydih, test_size=0.2)

    modeltot = fullModel(nbsteps)
    modeldih = fullModel(nbsteps)

    # totalcp_path = "cmd-cptot/cptot-{epoch:04d}.ckpt"
    # totalcp_dir = os.path.dirname(totalcp_path)
    # latest_totalcp = tf.train.latest_checkpoint(totalcp_dir)
    # modeltot = fullModel(nbsteps)
    # modeltot.load_weights(latest_totalcp)

    # dihedralcp_path = "cmd-cpdih/cpdih-{epoch:04d}.ckpt"
    # dihedralcp_dir = os.path.dirname(dihedralcp_path)
    # latest_dihedralcp = tf.train.latest_checkpoint(dihedralcp_dir)
    # modeldih = fullModel(nbsteps)
    # modeldih.load_weights(latest_dihedralcp)

    isExist = os.path.exists("equil-cptot")
    if not isExist:
        os.makedirs("equil-cptot")
    isExist = os.path.exists("equil-cpdih")
    if not isExist:
        os.makedirs("equil-cpdih")

    cptot_path = "equil-cptot/cptot-{epoch:04d}.ckpt"
    cpdih_path = "equil-cpdih/cpdih-{epoch:04d}.ckpt"

    cptot_callback = ModelCheckpoint(filepath=cptot_path, save_weights_only=True, save_freq="epoch")
    cpdih_callback = ModelCheckpoint(filepath=cpdih_path, save_weights_only=True, save_freq="epoch")

    historytot = modeltot.fit(Xtot_train, Ytot_train, epochs=50, batch_size=100, verbose=0,
                              callbacks=[cptot_callback], validation_data=[Xtot_test, Ytot_test])
    historydih = modeldih.fit(Xdih_train, Ydih_train, epochs=50, batch_size=100, verbose=0,
                              callbacks=[cpdih_callback], validation_data=[Xdih_test, Ydih_test])

    totadjustboost, dihadjustboost = 100, 0
    Ytot = modeltot.predict(Xtot)
    Ytot = np.absolute(totadjustboost + Ytot.flatten())
    Ydih = modeldih.predict(Xdih)
    Ydih = np.absolute(dihadjustboost + Ydih.flatten())
    Ydual = np.add(Ytot,Ydih)
    anharmtot, anharmdih, anharmdual = anharm(Ytot), anharm(Ydih), anharm(Ydual)

    # totalfc0s, dihedralfc0s, totalfscales, dihedralfscales = [], [], [], []
    if anharmdual <= 1e-2:
        plt.figure(figsize=(9, 6))
        sns.kdeplot(Ytot, color="Blue")
        sns.kdeplot(Ydih, color="Orange")
        sns.kdeplot(Ydual, color="Green")
        plt.legend(labels=["Total(" + str(round(anharmtot, 4)) + "; " + str(round(np.mean(Ytot), 3)) + "+/-" + str(round(np.std(Ytot), 3)) + ")",
                        "Dihedral(" + str(round(anharmdih, 4)) + "; " + str(round(np.mean(Ydih), 3)) + "+/-" + str(round(np.std(Ydih), 3)) + ")",
                        "Dual(" + str(round(anharmdual, 4)) + "; " + str(round(np.mean(Ydual), 3)) + "+/-" + str(round(np.std(Ydual), 3)) + ")"])
        plt.xlabel("Boost (kcal/mol)", fontsize=16, rotation=0)
        plt.ylabel("Frequency", fontsize=16, rotation=90)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig("equil-boosts/boostdistr" + str(iter + 1) + ".png")

        totalfc0s, dihedralfc0s, totalfscales, dihedralfscales = [], [], [], []
        for i in np.arange(0, len(Ydih), 100):
            dVD, VminD, VmaxD, VD = Ydih[i], min_dihedral, max_dihedral, Xdih[i]
            k0D_prime = ((sqrt(2*dVD*(VmaxD-VminD))-sqrt(2*dVD*(VmaxD-VminD)-4*(VminD-VD)*(VmaxD-VminD)))/(2*(VminD-VD)))**2
            k0D_doubleprime = ((sqrt(2*dVD*(VmaxD-VminD))+sqrt(2*dVD*(VmaxD-VminD)-4*(VminD-VD)*(VmaxD-VminD)))/(2*(VminD-VD)))**2
            k0D = min(k0D_prime, k0D_doubleprime)
            dihedralrefE = VminD + (VmaxD - VminD) / k0D
            if k0D >= 1:
                dihedralrefE = VmaxD
                k0D = (2*dVD*(VmaxD-VminD)) / (dihedralrefE - VD)**2
            kD = 1 - (k0D / (VmaxD - VminD)) * (dihedralrefE - VD)
            dihedralfc0s.append(k0D)
            dihedralfscales.append(kD)

        # for i in np.arange(0, len(Ytot), 100):
            dVP, VminP, VmaxP, VP = Ytot[i], min_total, max_total, Xtot[i]
            k0P_prime = ((sqrt(2*dVP*(VmaxP-VminP))-sqrt(2*dVP*(VmaxP-VminP)-4*(VminP-VP)*(VmaxP-VminP)))/(2*(VminP-VP)))**2
            k0P_doubleprime = ((sqrt(2*dVP*(VmaxP-VminP))+sqrt(2*dVP*(VmaxP-VminP)-4*(VminP-VP)*(VmaxP-VminP)))/(2*(VminP-VP)))**2
            k0P = min(k0P_prime, k0P_doubleprime)
            totalrefE = VminP + (VmaxP - VminP) / k0P
            if k0P >= 1:
                totalrefE = VmaxP
                k0P = (2*dVP*(VmaxP - VminP)) / (totalrefE - VP)**2
            kP = 1 - (k0P / (VmaxP - VminP)) * (totalrefE - VP)
            totalfc0s.append(k0P)
            totalfscales.append(kP)

        totalfc0s = np.asarray([totalfc0s]).flatten()
        dihedralfc0s = np.asarray([dihedralfc0s]).flatten()
        plt.figure(figsize=(9, 6))
        plt.scatter(np.arange(0, len(totalfc0s)), totalfc0s, label="Total k0P")
        plt.scatter(np.arange(0, len(dihedralfc0s)), dihedralfc0s, label="Dihedral k0D")
        plt.ylim(0, 1.5)
        plt.xlabel("Steps", fontsize=16, rotation=0)
        plt.ylabel("k0P / k0D", fontsize=16, rotation=90)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.savefig("equil-boosts/fconstants0" + str(iter + 1) + ".png")

        plt.figure(figsize=(9, 6))
        plt.scatter(Xtot, Ytot)
        plt.xlabel("Total Potentials (kcal/mol)", fontsize=16, rotation=0)
        plt.ylabel("Total Boosts (kcal/mol)", fontsize=16, rotation=90)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend()
        plt.savefig("equil-boosts/etotal" + str(iter + 1) + ".png")

        plt.figure(figsize=(9, 6))
        plt.scatter(Xdih, Ydih)
        plt.xlabel("Dihedral Potentials (kcal/mol)", fontsize=16, rotation=0)
        plt.ylabel("Dihedral Boosts (kcal/mol)", fontsize=16, rotation=90)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend()
        plt.savefig("equil-boosts/edihedral" + str(iter + 1) + ".png")

        isExist = os.path.exists("gamd-restart.dat")
        if isExist:
            os.remove("gamd-restart.dat")
        gamdRestart = open("gamd-restart.dat", "w")
        gamdRestart.write("#Parameters\tValues(kcal/mol)\n")
        gamdRestart.write("(0)Trained Steps:\t" + str(nbsteps) + "\n")
        gamdRestart.write("(1)Boosted VminD:\t" + str(min(dihedralEnergy[-10000:])) + "\n")
        gamdRestart.write("(2)Boosted VmaxD:\t" + str(max(dihedralEnergy[-10000:])) + "\n")
        gamdRestart.write("(3)Final k0D:\t" + str(dihedralfc0s[-1]) + "\n")
        gamdRestart.write("(4)Average k0D:\t" + str(np.mean([k0D for k0D in dihedralfc0s if np.isnan(k0D) == False])) + "\n")
        gamdRestart.write("(5)Maximum k0D:\t" + str(np.max([k0D for k0D in dihedralfc0s if np.isnan(k0D) == False])) + "\n")
        gamdRestart.write("(6)Boosted VminP:\t" + str(min(totalEnergy[-10000:])) + "\n")
        gamdRestart.write("(7)Boosted VmaxP:\t" + str(max(totalEnergy[-10000:])) + "\n")
        gamdRestart.write("(8)Final k0P:\t" + str(totalfc0s[-1]) + "\n")
        gamdRestart.write("(9)Average k0P:\t" + str(np.mean([k0P for k0P in totalfc0s if np.isnan(k0P) == False])) + "\n")
        gamdRestart.write("(10)Maximum k0P:\t" + str(np.max([k0P for k0P in totalfc0s if np.isnan(k0P) == False])) + "\n")
        gamdRestart.close()

    print("Iteration: " + str(iter + 1))
    print("The anharmonicity of total boosts after iteration " + str(iter + 1) + " is: " + str(anharmtot))
    print("The minimum and maximum total boost potential after iteration " + str(iter + 1) + " is: "
          + str(min(Ytot)) + " and " + str(max(Ytot)))
    print("The anharmonicity of dihedral boosts after iteration " + str(iter + 1) + " is: " + str(anharmdih))
    print("The minimum and maximum dihedral boost potential after iteration " + str(iter + 1) + " is: "
          + str(min(Ydih)) + " and " + str(max(Ydih)))
    print("The anharmonicity of dual boosts after iteration " + str(iter + 1) + " is: " + str(anharmdual))
    print("The minimum and maximum dual boost potential after iteration " + str(iter + 1) + " is: "
          + str(min(Ydual)) + " and " + str(max(Ydual)))

    iter += 1
