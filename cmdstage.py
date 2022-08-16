from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
from utils import *
import sys, os
"""
OpenMM Custom Integrator Manual: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
"""
class conventionalMDIntegrator(CustomIntegrator):
    def __init__(self, dt=0.002*picoseconds, temperature=300*kelvins):
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
        # self.addComputePerDof("v", "vscale*v + fscale*(f0+f2)/m + noisescale*gaussian/sqrt(m)")
        self.addComputePerDof("x", "x+dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "(x-oldx)/dt")
"""
OpenMM Application Manual: http://docs.openmm.org/7.2.0/userguide/application.html
"""
prmtop = AmberPrmtopFile("dip.top")
# inpcrd = AmberInpcrdFile("dip.crd")
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=8*angstrom, constraints=HBonds)
set_dihedral_group(system)
integrator = conventionalMDIntegrator(0.002*picoseconds, 300*kelvins)

isExist = os.path.exists("cmd.mdout")
if isExist:
    os.remove("cmd.mdout")
simulation = Simulation(prmtop.topology, system, integrator)
# simulation.context.setPositions(inpcrd.positions)
# if inpcrd.boxVectors is not None:
    # simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
# simulation.minimizeEnergy()
simulation.loadState("md-1ns.rst")
simulation.reporters.append(DCDReporter("cmd.dcd", 10))
simulation.reporters.append(ExpandedStateDataReporter(system, "cmd.mdout", 1, step=True, temperature=True,
                                                      brokenOutForceEnergies=True, potentialEnergy=True,
                                                      totalEnergy=True, density=True, separator="\t",
                                                      speed=True, remainingTime=True, totalSteps=1000000))
simulation.step(1000000)
simulation.saveState("cmd.rst")
