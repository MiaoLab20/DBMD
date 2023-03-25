import sys, os
from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from DLGaMDSimulationIntegrator import *
from utils import *
from simParams import *
"""
OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
"""
prmtop = AmberPrmtopFile(parmFile)
inpcrd = AmberInpcrdFile(crdFile)
if simType == "explicit": system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=nbCutoff*angstrom, constraints=HBonds)
else: system = prmtop.createSystem(implicitSolvent=GBn2, implicitSolventKappa=1.0/nanometer, soluteDielectric=1.0, solventDielectric=78.5)
set_dihedral_group(system)
integrator = conventionalMDIntegrator(0.002*picoseconds, temperature*kelvins)

isExist = os.path.exists("cmd.mdout")
if isExist:
    os.remove("cmd.mdout")
simulation = Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.minimizeEnergy()
simulation.reporters.append(DCDReporter("cmd.dcd", cmdRestartFreq))
simulation.reporters.append(ExpandedStateDataReporter(system, "cmd.mdout", cmdRestartFreq, step=True, temperature=True,
                                                      brokenOutForceEnergies=True, potentialEnergy=True,
                                                      totalEnergy=True, density=True, separator="\t",
                                                      speed=True, remainingTime=True, totalSteps=ntcmd))
simulation.step(ntcmd)
simulation.saveState("cmd.rst")
