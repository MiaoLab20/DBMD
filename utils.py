from openmm.app.statedatareporter import StateDataReporter
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np

class ExpandedStateDataReporter(StateDataReporter):
    def __init__(self, system, file, reportInterval, step=True, time=False, brokenOutForceEnergies=True,
                 potentialEnergy=True, kineticEnergy=False, totalEnergy=True,
                 temperature=True, volume=False, density=True, progress=False,
                 remainingTime=True, speed=True, elapsedTime=False, separator="\t",
                 systemMass=None, totalSteps=None):
        self._brokenOutForceEnergies = brokenOutForceEnergies
        self._system = system
        # use super() to inherit methods from parent class StateDataReporter
        super().__init__(file, reportInterval, step, time, potentialEnergy, kineticEnergy, totalEnergy,
                         temperature, volume, density, progress, remainingTime, speed, elapsedTime, separator,
                         systemMass, totalSteps)

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._brokenOutForceEnergies:
            # use enumerate to print out both the count and value of item of current iteration
            for i, force in enumerate(self._system.getForces()):
                values.append(simulation.context.getState(
                    getEnergy=True,
                    groups={i}).getPotentialEnergy().value_in_unit(
                        kilojoules_per_mole))
        return values

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._brokenOutForceEnergies:
            # use enumerate to print out both the count and value of item of current iteration
            for i, force in enumerate(self._system.getForces()):
                headers.append(force.__class__.__name__)
        return headers

def set_all_forces_to_group(system):
    group = 1
    for force in system.getForces():
        force.setForceGroup(group)
    return group

def set_single_group(group, name, system):
    for force in system.getForces():
        if force.__class__.__name__ == name:
            force.setForceGroup(group)
            break
    return group

def set_dihedral_group(system):
    return set_single_group(2, 'PeriodicTorsionForce', system)

def set_non_bonded_group(system):
    return set_single_group(1, 'NonbondedForce', system)

def anharm(data):
    var = np.var(data)
    hist, edges = np.histogram(data, 50, density=True)
    hist = np.add(hist,0.000000000000000001)
    dx = edges[1] - edges[0]
    S1 = (-1) * np.trapz(np.multiply(hist, np.log(hist)),dx=dx)
    S2 = (1/2) * np.log(np.add(2.00*np.pi*np.exp(1)*var,0.000000000000000001))
    return (S2 - S1)
